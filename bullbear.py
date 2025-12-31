# =========================
# Part 1/7 — bullbear.py
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import time
import pytz

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
  #MainMenu, header, footer {visibility: hidden;}
  .stTabs [data-baseweb="tab-list"] { gap: 18px; }
  .stTabs [data-baseweb="tab"] { height: 48px; }
</style>
""",
    unsafe_allow_html=True,
)

PACIFIC = pytz.timezone("US/Pacific")
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

# ---------------------------
# Auto-refresh (PST)
# ---------------------------
REFRESH_INTERVAL = 120  # seconds


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
# Session state init
# ---------------------------
if "asset_mode" not in st.session_state:
    st.session_state.asset_mode = "Forex"  # default
if "last_run_params" not in st.session_state:
    st.session_state.last_run_params = None  # dict with ticker/view/hour_range/mode


def _reset_run_state_for_mode_switch():
    st.session_state.last_run_params = None


# ---------------------------
# Top header + mode buttons
# ---------------------------
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
# Helpers (format + styling)
# ---------------------------


def style_axes(ax):
    try:
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    except Exception:
        pass


def _coerce_1d_series(obj) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        # Prefer a column named Close if present; else first numeric column
        if "Close" in obj.columns:
            col = obj["Close"]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            s = col
        else:
            num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
            if not num_cols:
                return pd.Series(dtype=float)
            col = obj[num_cols[0]]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            s = col
    else:
        try:
            s = pd.Series(obj)
        except Exception:
            return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")


def fmt_price_val(y: float) -> str:
    try:
        y = float(y)
    except Exception:
        return "n/a"
    return f"{y:,.5f}" if abs(y) < 10 else f"{y:,.3f}"


def fmt_slope(m: float) -> str:
    try:
        mv = float(np.squeeze(m))
    except Exception:
        return "n/a"
    return f"{mv:.4f}" if np.isfinite(mv) else "n/a"


def fmt_pct(x, digits: int = 1) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"


def fmt_r2(r2: float, digits: int = 1) -> str:
    try:
        rv = float(r2)
    except Exception:
        return "n/a"
    return fmt_pct(rv, digits=digits) if np.isfinite(rv) else "n/a"


def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01,
        y_val,
        text,
        transform=trans,
        ha="left",
        va="center",
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        zorder=6,
    )


def subset_by_daily_view(obj, view_label: str):
    if obj is None:
        return obj
    if hasattr(obj, "empty") and obj.empty:
        return obj
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj
    idx = obj.index
    end = idx.max()
    days_map = {"6M": 182, "12M": 365, "24M": 730}
    if view_label == "Historical":
        start = idx.min()
    else:
        start = end - pd.Timedelta(days=days_map.get(view_label, 365))
    return obj.loc[(idx >= start) & (idx <= end)]


def pip_size_for_symbol(symbol: str):
    s = str(symbol).upper()
    if "=X" not in s:
        return None
    return 0.01 if "JPY" in s else 0.0001


def _diff_text(a: float, b: float, symbol: str) -> str:
    try:
        av = float(a)
        bv = float(b)
    except Exception:
        return ""
    ps = pip_size_for_symbol(symbol)
    diff = abs(bv - av)
    if ps:
        return f"{diff/ps:.1f} pips"
    return f"Δ {diff:.3f}"


ALERT_TEXT = (
    "ALERT: Trend may be changing - Open trade position with caution while still following the signals on the chat."
)


def format_trade_instruction(
    trend_slope: float,
    buy_val: float,
    sell_val: float,
    close_val: float,
    symbol: str,
    global_trend_slope: float = None,
) -> str:
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    if global_trend_slope is None:
        try:
            uptrend = float(trend_slope) >= 0.0
        except Exception:
            uptrend = True
        if uptrend:
            a, b = entry_buy, exit_sell
            t = f"▲ BUY @{fmt_price_val(a)} → ▼ SELL @{fmt_price_val(b)}"
        else:
            a, b = exit_sell, entry_buy
            t = f"▼ SELL @{fmt_price_val(a)} → ▲ BUY @{fmt_price_val(b)}"
        return t + f" • {_diff_text(a, b, symbol)}"

    try:
        g = float(global_trend_slope)
        l = float(trend_slope)
    except Exception:
        return ALERT_TEXT

    if not (np.isfinite(g) and np.isfinite(l)) or g == 0 or l == 0:
        return ALERT_TEXT

    if g > 0 and l > 0:
        a, b = entry_buy, exit_sell
        return f"▲ BUY @{fmt_price_val(a)} → ▼ SELL @{fmt_price_val(b)} • {_diff_text(a, b, symbol)}"
    if g < 0 and l < 0:
        a, b = exit_sell, entry_buy
        return f"▼ SELL @{fmt_price_val(a)} → ▲ BUY @{fmt_price_val(b)} • {_diff_text(a, b, symbol)}"
    return ALERT_TEXT


def hour_range_to_yf_period(hour_range: str) -> str:
    m = {"24h": "1d", "48h": "2d", "72h": "3d", "7d": "7d", "14d": "14d", "30d": "30d", "60d": "60d"}
    return m.get(hour_range, "1d")
# =========================
# Part 2/7 — bullbear.py
# =========================
# ---------------------------
# Sidebar configuration
# ---------------------------
st.sidebar.title("Configuration")

if st.sidebar.button("🧹 Clear cache (data + run state)", use_container_width=True, key="btn_clear_cache"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state.last_run_params = None
    try:
        st.experimental_rerun()
    except Exception:
        pass

bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly)", value=True, key="sb_show_fibs")
show_fibs_daily = st.sidebar.checkbox("Show Fibonacci (daily)", value=True, key="sb_show_fibs_daily")

slope_lb_daily = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
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
show_ntd_panel = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_ntd_panel")

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max = st.sidebar.slider("PSAR max", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

st.sidebar.subheader("Signal Logic")
signal_threshold = st.sidebar.slider("Confidence threshold (R²)", 0.50, 0.999, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

st.sidebar.subheader("NTD / NPX")
show_ntd = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window = st.sidebar.slider("NTD/NPX window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")
show_npx_ntd = st.sidebar.checkbox("Overlay NPX on NTD", value=True, key="sb_show_npx_ntd")
mark_npx_cross = st.sidebar.checkbox("Mark NPX 0.0-cross", value=True, key="sb_mark_npx_cross")

st.sidebar.subheader("NTD Channel (Hourly)")
show_ntd_channel = st.sidebar.checkbox("Highlight when price is between S/R on NTD", value=True, key="sb_ntd_channel")

st.sidebar.subheader("Ichimoku Kijun")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Tenkan", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Kijun", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema = st.sidebar.checkbox("Use EMA midline", value=False, key="sb_bb_ema")

st.sidebar.subheader("HMA(55)")
show_hma = st.sidebar.checkbox("Show HMA", value=True, key="sb_hma_show")
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")

st.sidebar.subheader("HMA reversal markers on NTD")
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA reversal on NTD", value=True, key="sb_hma_rev_ntd")
hma_rev_lb = st.sidebar.slider("HMA reversal lookback", 2, 10, 3, 1, key="sb_hma_rev_lb")

if mode == "Forex":
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_sessions_pst = False

# Universe
if mode == "Stock":
    universe = sorted(
        [
            "AAPL",
            "SPY",
            "AMZN",
            "DIA",
            "TSLA",
            "SPGI",
            "JPM",
            "VTWG",
            "PLTR",
            "NVDA",
            "META",
            "SITM",
            "MARA",
            "GOOG",
            "HOOD",
            "BABA",
            "IBM",
            "AVGO",
            "GUSH",
            "VOO",
            "MSFT",
            "TSM",
            "NFLX",
            "MP",
            "AAL",
            "URI",
            "DAL",
            "BBAI",
            "QUBT",
            "AMD",
            "SMCI",
            "ORCL",
            "TLT",
        ]
    )
else:
    universe = [
        "EURUSD=X",
        "EURJPY=X",
        "GBPUSD=X",
        "USDJPY=X",
        "AUDUSD=X",
        "NZDUSD=X",
        "CADJPY=X",
        "HKDJPY=X",
        "USDCAD=X",
        "USDCNY=X",
        "USDCHF=X",
        "EURGBP=X",
        "EURCAD=X",
        "NZDJPY=X",
        "USDHKD=X",
        "EURHKD=X",
        "GBPHKD=X",
        "GBPJPY=X",
        "CNHJPY=X",
        "AUDJPY=X",
        "GBPCAD=X",
    ]
# =========================
# Part 3/7 — bullbear.py
# =========================
# ---------------------------
# Data fetchers + gapless
# ---------------------------

def _normalize_yf_single_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns like ('Close','AAPL').
    For a single ticker, drop the ticker level -> columns: Open/High/Low/Close/...
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels >= 2:
        lvl1 = df.columns.get_level_values(1)
        if len(pd.Index(lvl1).unique()) == 1:
            df = df.copy()
            df.columns = df.columns.droplevel(1)
    return df


def make_gapless_ohlc(
    df: pd.DataFrame,
    price_cols=("Open", "High", "Low", "Close"),
    gap_mult: float = 12.0,
    min_gap_seconds: float = 3600.0,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if "Close" not in df.columns:
        return df

    ref_col = "Open" if "Open" in df.columns else "Close"
    close = pd.to_numeric(df["Close"], errors="coerce")
    refp = pd.to_numeric(df[ref_col], errors="coerce")
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
            dt_sec = float((idx[i] - idx[i - 1]).total_seconds())
        except Exception:
            dt_sec = 0.0

        if dt_sec >= thr:
            prev_close = float(close.iloc[i - 1]) if np.isfinite(close.iloc[i - 1]) else np.nan
            curr_ref = float(refp.iloc[i]) if np.isfinite(refp.iloc[i]) else np.nan
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
def fetch_hist_close(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
    df = _normalize_yf_single_ticker_columns(df)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" not in df.columns:
        # fallback: first numeric col
        s = _coerce_1d_series(df)
    else:
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        s = pd.to_numeric(c, errors="coerce")

    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)

    s.index = pd.to_datetime(s.index)
    s = s.asfreq("D").ffill()

    if isinstance(s.index, pd.DatetimeIndex):
        if s.index.tz is None:
            s = s.tz_localize(PACIFIC)
        else:
            s = s.tz_convert(PACIFIC)
    return s


@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
    df = _normalize_yf_single_ticker_columns(df)
    if df is None or df.empty:
        return pd.DataFrame()

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(how="all")

    df.index = pd.to_datetime(df.index)
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df = df.tz_localize(PACIFIC)
        else:
            df = df.tz_convert(PACIFIC)
    return df


@st.cache_data(ttl=120)
def fetch_intraday_5m(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m", progress=False, auto_adjust=False)
    df = _normalize_yf_single_ticker_columns(df)
    if df is None or df.empty:
        return df

    df.index = pd.to_datetime(df.index)
    # yfinance intraday is usually UTC-naive; treat as UTC then convert to PST
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        df = df.tz_convert(PACIFIC)

    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)
    return df


@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        idx = pd.date_range(datetime.now(tz=PACIFIC).date(), periods=30, freq="D", tz=PACIFIC)
        return idx, pd.Series([np.nan] * 30, index=idx), pd.DataFrame(index=idx)

    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)

    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    except Exception:
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()


def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return {}
    hi = float(s.max())
    lo = float(s.min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5 * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
        "100%": lo,
    }


def annotate_fib_extreme_warning(
    ax,
    close: pd.Series,
    fibs: dict,
    r2_val: float,
    min_r2: float = 0.99,
    tol_frac: float = 0.0015,
    lookback_bars: int = 5,
):
    try:
        r2 = float(r2_val)
    except Exception:
        return
    if not np.isfinite(r2) or r2 < float(min_r2):
        return

    c = _coerce_1d_series(close).dropna()
    if c.empty or "0%" not in fibs or "100%" not in fibs:
        return
    hi, lo = fibs["0%"], fibs["100%"]
    n = int(min(len(c), max(1, lookback_bars)))
    tail_idx = list(c.index[-n:])

    def _tol(px: float) -> float:
        return max(abs(px) * float(tol_frac), 1e-9)

    for t in reversed(tail_idx):
        px = float(c.loc[t])
        if abs(px - hi) <= _tol(px):
            ax.scatter([t], [px], marker="D", s=120, color="tab:orange", zorder=20, label="Fib 0% warning")
            ax.text(t, px, "  ⚠ Fib 0% possible reversal (R²≥0.99)", fontsize=8, color="tab:orange", va="bottom", zorder=20)
            break
        if abs(px - lo) <= _tol(px):
            ax.scatter([t], [px], marker="D", s=120, color="tab:orange", zorder=20, label="Fib 100% warning")
            ax.text(t, px, "  ⚠ Fib 100% possible reversal (R²≥0.99)", fontsize=8, color="tab:orange", va="top", zorder=20)
            break
# =========================
# Part 4/7 — bullbear.py
# =========================
# ---------------------------
# Indicators + math
# ---------------------------
def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
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
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2


def slope_reversal_probability(
    series_like, current_slope: float, hist_window: int = 240, slope_window: int = 60, horizon: int = 15
) -> float:
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


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    return macd.reindex(s.index), sig.reindex(s.index), hist.reindex(s.index)


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
    ax.plot(s.index, yhat, linestyle="--", linewidth=2.4, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return float(m)


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


def _cross_level(series: pd.Series, level: float = 0.0):
    s = _coerce_1d_series(series)
    prev = s.shift(1)
    cross_up = (s >= level) & (prev < level)
    cross_dn = (s <= level) & (prev > level)
    return cross_up.fillna(False), cross_dn.fillna(False)
# =========================
# Part 5/7 — bullbear.py
# =========================
# ---------------------------
# Supertrend + PSAR + Ichimoku + NTD helpers + Stars + Sessions
# ---------------------------

def _compute_atr_from_ohlc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.Series(dtype=float)
    high = _coerce_1d_series(df["High"])
    low = _coerce_1d_series(df["Low"])
    close = _coerce_1d_series(df["Close"])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr.reindex(df.index)


def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame(columns=["ST", "in_uptrend"])
    ohlc = df[["High", "Low", "Close"]].copy()
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

        if ohlc["Close"].iloc[i] > upperband.iloc[i - 1]:
            in_uptrend.iloc[i] = True
        elif ohlc["Close"].iloc[i] < lowerband.iloc[i - 1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i - 1]
            if in_uptrend.iloc[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if (not in_uptrend.iloc[i]) and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        st_line.iloc[i] = lowerband.iloc[i] if in_uptrend.iloc[i] else upperband.iloc[i]

    return pd.DataFrame({"ST": st_line, "in_uptrend": in_uptrend})


def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        return pd.DataFrame(columns=["PSAR", "in_uptrend"])
    high = _coerce_1d_series(df["High"])
    low = _coerce_1d_series(df["Low"])
    idx = high.index.union(low.index)
    high = high.reindex(idx)
    low = low.reindex(idx)

    psar = pd.Series(index=idx, dtype=float)
    in_uptrend = pd.Series(index=idx, dtype=bool)

    in_uptrend.iloc[0] = True
    psar.iloc[0] = float(low.iloc[0])
    ep = float(high.iloc[0])
    af = step

    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i - 1]
        if in_uptrend.iloc[i - 1]:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i], float(low.iloc[i - 1]), float(low.iloc[i - 2]) if i >= 2 else float(low.iloc[i - 1]))
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
            psar.iloc[i] = max(psar.iloc[i], float(high.iloc[i - 1]), float(high.iloc[i - 2]) if i >= 2 else float(high.iloc[i - 1]))
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


def ichimoku_kijun(high: pd.Series, low: pd.Series, base: int = 26):
    h = _coerce_1d_series(high)
    l = _coerce_1d_series(low)
    kijun = ((h.rolling(base).max() + l.rolling(base).min()) / 2.0)
    return kijun


def overlay_inrange_on_ntd(ax, price: pd.Series, sup: pd.Series, res: pd.Series):
    p = _coerce_1d_series(price)
    s = _coerce_1d_series(sup).reindex(p.index).ffill()
    r = _coerce_1d_series(res).reindex(p.index).ffill()
    ok = p.notna() & s.notna() & r.notna()
    if ok.sum() < 3:
        return
    p = p[ok]
    s = s[ok]
    r = r[ok]
    inrange = (p >= s) & (p <= r)
    if not inrange.any():
        return
    ax.fill_between(p.index, -1.05, 1.05, where=inrange.values, alpha=0.06, step=None, label="Price in S↔R")


def overlay_npx_on_ntd(ax, npx: pd.Series, trend_slope: float, mark_crosses: bool = True):
    x = _coerce_1d_series(npx)
    if x.dropna().empty:
        return
    ax.plot(x.index, x.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX (Norm Price)")
    if not mark_crosses:
        return
    try:
        ts = float(trend_slope)
    except Exception:
        return
    if not np.isfinite(ts) or ts == 0.0:
        return
    cross_up0, cross_dn0 = _cross_level(x, 0.0)
    if ts > 0:
        idx_up = list(cross_up0[cross_up0].index)
        if idx_up:
            ax.scatter(idx_up, x.loc[idx_up], marker="o", s=40, color="tab:green", zorder=9, label="NPX 0↑ (Uptrend)")
    else:
        idx_dn = list(cross_dn0[cross_dn0].index)
        if idx_dn:
            ax.scatter(idx_dn, x.loc[idx_dn], marker="x", s=60, color="tab:red", zorder=9, label="NPX 0↓ (Downtrend)")


def overlay_npx_zero_cross_triangles(ax, npx: pd.Series, trend_slope: float):
    s = _coerce_1d_series(npx).dropna()
    if s.empty:
        return
    try:
        ts = float(trend_slope)
    except Exception:
        return
    if not np.isfinite(ts) or ts == 0.0:
        return
    uptrend = ts > 0
    downtrend = ts < 0
    cross_up0, cross_dn0 = _cross_level(s, 0.0)
    idx_up0 = list(cross_up0[cross_up0].index)
    idx_dn0 = list(cross_dn0[cross_dn0].index)
    if uptrend and idx_dn0:
        ax.scatter(idx_dn0, s.loc[idx_dn0], marker="v", s=85, color="tab:green", zorder=10, label="NPX 0↓ (Uptrend)")
    if downtrend and idx_up0:
        ax.scatter(idx_up0, s.loc[idx_up0], marker="^", s=85, color="tab:red", zorder=10, label="NPX 0↑ (Downtrend)")


def hma_npx_star_masks(close: pd.Series, hma: pd.Series, npx: pd.Series, max_bar_gap: int = 2):
    c = _coerce_1d_series(close).astype(float)
    h = _coerce_1d_series(hma).reindex(c.index)
    x = _coerce_1d_series(npx).reindex(c.index)
    ok = c.notna() & h.notna() & x.notna()
    if ok.sum() < 4:
        idx = c.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)

    c = c[ok]
    h = h[ok]
    x = x[ok]
    cross_hma_up, cross_hma_dn = _cross_series(c, h)

    npx_up_m05, _ = _cross_level(x, -0.5)
    _, npx_dn_p05 = _cross_level(x, +0.5)

    npx_up_m05 = npx_up_m05 & (x.diff() > 0)
    npx_dn_p05 = npx_dn_p05 & (x.diff() < 0)

    w = int(2 * max(1, int(max_bar_gap)) + 1)
    near_npx_up = npx_up_m05.rolling(w, center=True, min_periods=1).max().fillna(False).astype(bool)
    near_npx_dn = npx_dn_p05.rolling(w, center=True, min_periods=1).max().fillna(False).astype(bool)

    price_up = (c.diff() > 0).fillna(False)
    price_dn = (c.diff() < 0).fillna(False)
    npx_up_now = (x.diff() > 0).fillna(False)
    npx_dn_now = (x.diff() < 0).fillna(False)

    buy_mask = cross_hma_up & near_npx_up & price_up & npx_up_now
    sell_mask = cross_hma_dn & near_npx_dn & price_dn & npx_dn_now

    buy_mask = buy_mask.reindex(_coerce_1d_series(close).index, fill_value=False)
    sell_mask = sell_mask.reindex(_coerce_1d_series(close).index, fill_value=False)
    return buy_mask, sell_mask


def find_slope_trigger_after_band_reversal(
    price: pd.Series, yhat: pd.Series, upper_band: pd.Series, lower_band: pd.Series, horizon: int = 15
):
    p = _coerce_1d_series(price)
    y = _coerce_1d_series(yhat).reindex(p.index)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & y.notna() & u.notna() & l.notna()
    if ok.sum() < 3:
        return None
    p = p[ok]
    y = y[ok]
    u = u[ok]
    l = l[ok]

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
        window = touch_mask.iloc[j0 : loc + 1]
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
    ax.annotate("", xy=(t1, p1), xytext=(t0, p0), arrowprops=dict(arrowstyle="->", color=col, lw=2.0, alpha=0.85), zorder=9)
    ax.scatter([t1], [p1], marker="o", s=90, color=col, zorder=10, label=lbl)
    ax.text(t1, p1, f"  {lbl}", color=col, fontsize=9, fontweight="bold", va="bottom" if side == "BUY" else "top", zorder=10)


def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes
    start_d = idx[0].astimezone(session_tz).date()
    end_d = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
        dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))
        dt_open_pst = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)
    return opens, closes


def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close = session_markers_for_index(idx, NY_TZ, 8, 17)
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
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=1.6, label="London Open"),
        Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=1.6, label="New York Open"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.6, label="New York Close"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels
# =========================
# Part 6/7 — bullbear.py
# =========================
# ---------------------------
# Renderers (Daily + Hourly)
# ---------------------------

def render_daily(sel: str, show_forecast: bool = True):
    ohlc = fetch_hist_ohlc(sel)
    if ohlc is None or ohlc.empty:
        st.warning("No daily data available.")
        return

    close_full = _coerce_1d_series(ohlc.get("Close")).dropna()
    ohlc_show = subset_by_daily_view(ohlc, daily_view)
    close = _coerce_1d_series(ohlc_show.get("Close")).dropna()
    if close.empty:
        st.warning("No daily data in selected view.")
        return

    res = close.rolling(sr_lb_daily, min_periods=1).max()
    sup = close.rolling(sr_lb_daily, min_periods=1).min()

    hma = compute_hma(close, period=hma_period)
    ntd = compute_normalized_trend(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)
    npx = compute_normalized_price(close, window=ntd_window) if show_npx_ntd else pd.Series(index=close.index, dtype=float)

    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    st_line = pd.Series(index=close.index, dtype=float)
    if {"High", "Low", "Close"}.issubset(ohlc_show.columns):
        st_df = compute_supertrend(ohlc_show, atr_period=atr_period, atr_mult=atr_mult)
        if not st_df.empty and "ST" in st_df.columns:
            st_line = st_df["ST"].reindex(close.index)

    psar_df = pd.DataFrame()
    if show_psar and {"High", "Low"}.issubset(ohlc_show.columns):
        psar_df = compute_psar_from_ohlc(ohlc_show, step=psar_step, max_step=psar_max)
        if not psar_df.empty:
            psar_df = psar_df.reindex(close.index)

    kijun = pd.Series(index=close.index, dtype=float)
    if show_ichi and {"High", "Low"}.issubset(ohlc_show.columns):
        kijun = ichimoku_kijun(ohlc_show["High"], ohlc_show["Low"], base=ichi_base).reindex(close.index)

    yhat, up2, lo2, m, r2 = regression_with_band(close, lookback=slope_lb_daily, z=2.0)

    npx_star = compute_normalized_price(close, window=ntd_window)
    buy_star, sell_star = hma_npx_star_masks(close, hma, npx_star, max_bar_gap=2)

    if show_ntd:
        fig, (ax, ax_ntd) = plt.subplots(2, 1, sharex=True, figsize=(14, 7), gridspec_kw={"height_ratios": [3.2, 1.3]})
        plt.subplots_adjust(hspace=0.05, top=0.90, right=0.78, bottom=0.22)
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax_ntd = None
        plt.subplots_adjust(top=0.88, right=0.78, bottom=0.22)

    ax.plot(close.index, close.values, label="Daily Close")
    global_m = draw_trend_direction_line(ax, close, label_prefix="Trend (global)")

    if show_hma and not hma.dropna().empty:
        ax.plot(hma.index, hma.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_bbands and not bb_up.dropna().empty:
        ax.fill_between(close.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    if show_ichi and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun.values, "-", linewidth=1.8, color="black", label=f"Kijun ({ichi_base})")

    if not st_line.dropna().empty:
        ax.plot(st_line.index, st_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        up_mask = psar_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_df.index[up_mask], psar_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6, label="PSAR")
        if dn_mask.any():
            ax.scatter(psar_df.index[dn_mask], psar_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    res_val = float(res.iloc[-1])
    sup_val = float(sup.iloc[-1])
    px_val = float(close.iloc[-1])
    ax.hlines(res_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
    ax.hlines(sup_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
    label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
    label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2, label=f"Slope {slope_lb_daily} ({fmt_slope(m)}/bar)")
    if not up2.empty and not lo2.empty:
        ax.plot(up2.index, up2.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax.plot(lo2.index, lo2.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")
        trig = find_slope_trigger_after_band_reversal(close, yhat, up2, lo2, horizon=rev_horizon)
        annotate_slope_trigger(ax, trig)

    if buy_star.any():
        idxb = list(buy_star[buy_star].index)
        ax.scatter(idxb, close.loc[idxb], marker="*", s=220, color="blue", zorder=14, label="BLUE ★ (HMA↑ + NPX -0.5↑)")
    if sell_star.any():
        idxs = list(sell_star[sell_star].index)
        ax.scatter(idxs, close.loc[idxs], marker="*", s=220, color="purple", zorder=14, label="PURPLE ★ (HMA↓ + NPX +0.5↓)")

    if show_fibs_daily:
        fibs = fibonacci_levels(close)
        for lbl, y in fibs.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dotted", linewidth=1)
            ax.text(close.index[-1], y, f" {lbl}", va="center")
        annotate_fib_extreme_warning(ax, close, fibs, r2, min_r2=0.99)

    nbb_txt = ""
    try:
        last_pct = float(bb_pctb.dropna().iloc[-1]) if show_bbands else np.nan
        last_nbb = float(bb_nbb.dropna().iloc[-1]) if show_bbands else np.nan
        if np.isfinite(last_nbb) and np.isfinite(last_pct):
            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
    except Exception:
        pass

    ax.text(
        0.99,
        0.02,
        f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7),
    )

    instr_txt = format_trade_instruction(
        trend_slope=m, buy_val=sup_val, sell_val=res_val, close_val=px_val, symbol=sel, global_trend_slope=global_m
    )
    rev_prob = slope_reversal_probability(close, m, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon)
    ax.set_title(f"{sel} Daily ({daily_view})  —  {instr_txt}  [P(slope rev≤{rev_horizon} bars)={fmt_pct(rev_prob)}]")

    if ax_ntd is not None:
        if show_ntd and shade_ntd and not _coerce_1d_series(ntd).dropna().empty:
            shade_ntd_regions(ax_ntd, ntd)
        if show_ntd and not _coerce_1d_series(ntd).dropna().empty:
            ax_ntd.plot(ntd.index, ntd.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
        if show_npx_ntd and not _coerce_1d_series(npx).dropna().empty:
            overlay_npx_zero_cross_triangles(ax_ntd, npx, trend_slope=m)
            overlay_npx_on_ntd(ax_ntd, npx, trend_slope=m, mark_crosses=mark_npx_cross)

        ax_ntd.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax_ntd.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        ax_ntd.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        ax_ntd.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        ax_ntd.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
        ax_ntd.set_ylim(-1.1, 1.1)
        ax_ntd.legend(loc="lower left", framealpha=0.5, fontsize=9)
        ax_ntd.set_xlabel("Time (PST)")
        style_axes(ax_ntd)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.5, fontsize=9, borderaxespad=0.0)
    style_axes(ax)
    st.pyplot(fig)

    if show_forecast:
        st.subheader("Forecast (SARIMAX)")
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close_full)
        figf, axf = plt.subplots(figsize=(14, 3.8))
        axf.plot(close_full.index[-180:], close_full.values[-180:], label="History (last ~180d)")
        axf.plot(fc_idx, fc_vals.values, label="Forecast")
        if isinstance(fc_ci, pd.DataFrame) and fc_ci.shape[1] >= 2:
            axf.fill_between(fc_idx, fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values, alpha=0.15, label="Conf. interval")
        axf.set_title(f"{sel} — 30D Forecast")
        axf.legend(loc="upper left", framealpha=0.5)
        style_axes(axf)
        st.pyplot(figf)


def render_hourly(sel: str, hour_range_label: str):
    period = hour_range_to_yf_period(hour_range_label)
    intraday = fetch_intraday_5m(sel, period=period)
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None

    plot_df = intraday.copy()
    plot_df.index = pd.RangeIndex(len(plot_df))
    intr = plot_df

    hc = _coerce_1d_series(intr["Close"]).ffill()
    if hc.dropna().empty:
        st.warning("No intraday close values.")
        return

    res = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup = hc.rolling(sr_lb_hourly, min_periods=1).min()
    hma = compute_hma(hc, period=hma_period)
    npx_star = compute_normalized_price(hc, window=ntd_window)

    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    st_line = pd.Series(index=hc.index, dtype=float)
    if {"High", "Low", "Close"}.issubset(intr.columns):
        st_df = compute_supertrend(intr, atr_period=atr_period, atr_mult=atr_mult)
        if not st_df.empty and "ST" in st_df.columns:
            st_line = st_df["ST"].reindex(hc.index)

    psar_df = pd.DataFrame()
    if show_psar and {"High", "Low"}.issubset(intr.columns):
        psar_df = compute_psar_from_ohlc(intr, step=psar_step, max_step=psar_max)
        if not psar_df.empty:
            psar_df = psar_df.reindex(hc.index)

    yhat, up2, lo2, m, r2 = regression_with_band(hc, lookback=slope_lb_hourly, z=2.0)
    rev_prob = slope_reversal_probability(hc, m, hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon)

    buy_star, sell_star = hma_npx_star_masks(hc, hma, npx_star, max_bar_gap=2)

    if show_ntd_panel:
        fig, (ax, ax_ntd) = plt.subplots(2, 1, sharex=True, figsize=(14, 7), gridspec_kw={"height_ratios": [3.2, 1.3]})
        plt.subplots_adjust(hspace=0.05, top=0.90, right=0.78, bottom=0.22)
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax_ntd = None
        plt.subplots_adjust(top=0.88, right=0.78, bottom=0.22)

    ax.plot(hc.index, hc.values, label="Intraday Close")
    global_m = draw_trend_direction_line(ax, hc, label_prefix="Trend (global)")

    if show_hma and not hma.dropna().empty:
        ax.plot(hma.index, hma.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_bbands and not bb_up.dropna().empty:
        ax.fill_between(hc.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    if not st_line.dropna().empty:
        ax.plot(st_line.index, st_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        up_mask = psar_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_df.index[up_mask], psar_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6, label="PSAR")
        if dn_mask.any():
            ax.scatter(psar_df.index[dn_mask], psar_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    res_val = float(res.iloc[-1])
    sup_val = float(sup.iloc[-1])
    px_val = float(hc.iloc[-1])
    ax.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
    ax.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
    label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
    label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} ({fmt_slope(m)}/bar)")
    if not up2.empty and not lo2.empty:
        ax.plot(up2.index, up2.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax.plot(lo2.index, lo2.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")
        trig = find_slope_trigger_after_band_reversal(hc, yhat, up2, lo2, horizon=rev_horizon)
        annotate_slope_trigger(ax, trig)

    if buy_star.any():
        idxb = list(buy_star[buy_star].index)
        ax.scatter(idxb, hc.loc[idxb], marker="*", s=220, color="blue", zorder=14, label="BLUE ★ (HMA↑ + NPX -0.5↑)")
    if sell_star.any():
        idxs = list(sell_star[sell_star].index)
        ax.scatter(idxs, hc.loc[idxs], marker="*", s=220, color="purple", zorder=14, label="PURPLE ★ (HMA↓ + NPX +0.5↓)")

    if show_fibs:
        fibs = fibonacci_levels(hc)
        for lbl, y in fibs.items():
            ax.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
            ax.text(hc.index[-1], y, f" {lbl}", va="center")
        annotate_fib_extreme_warning(ax, hc, fibs, r2, min_r2=0.99)

    session_handles = session_labels = None
    if mode == "Forex" and show_sessions_pst and isinstance(real_times, pd.DatetimeIndex) and not real_times.empty:
        sess = compute_session_lines(real_times)

        def _map_to_pos(times):
            if not times:
                return []
            t = pd.to_datetime(times)
            idxer = real_times.get_indexer(t, method="nearest")
            return [int(i) for i in idxer if 0 <= int(i) < len(hc)]

        sess_pos = {k: _map_to_pos(v) for k, v in sess.items()}
        session_handles, session_labels = draw_session_lines(ax, sess_pos)

    instr_txt = format_trade_instruction(m, sup_val, res_val, px_val, sel, global_trend_slope=global_m)
    ax.set_title(f"{sel} Intraday ({hour_range_label}) — {instr_txt}  [P(slope rev≤{rev_horizon} bars)={fmt_pct(rev_prob)}]")

    nbb_txt = ""
    try:
        last_pct = float(bb_pctb.dropna().iloc[-1]) if show_bbands else np.nan
        last_nbb = float(bb_nbb.dropna().iloc[-1]) if show_bbands else np.nan
        if np.isfinite(last_nbb) and np.isfinite(last_pct):
            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
    except Exception:
        pass
    ax.text(
        0.99,
        0.02,
        f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7),
    )

    if ax_ntd is not None:
        ntd = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        ax_ntd.set_title(f"Hourly Indicator Panel — NTD + NPX (S/R w={sr_lb_hourly})")
        if show_ntd and shade_ntd and not _coerce_1d_series(ntd).dropna().empty:
            shade_ntd_regions(ax_ntd, ntd)
        if show_ntd and not _coerce_1d_series(ntd).dropna().empty:
            ax_ntd.plot(ntd.index, ntd.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
        if show_ntd_channel:
            overlay_inrange_on_ntd(ax_ntd, price=hc, sup=sup, res=res)
        if show_npx_ntd and not _coerce_1d_series(npx).dropna().empty:
            overlay_npx_zero_cross_triangles(ax_ntd, npx, trend_slope=m)
            overlay_npx_on_ntd(ax_ntd, npx, trend_slope=m, mark_crosses=mark_npx_cross)

        ax_ntd.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax_ntd.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        ax_ntd.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        ax_ntd.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        ax_ntd.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
        ax_ntd.set_ylim(-1.1, 1.1)
        ax_ntd.legend(loc="lower left", framealpha=0.5, fontsize=9)
        ax_ntd.set_xlabel("Time (PST)")
        style_axes(ax_ntd)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.5, fontsize=9, borderaxespad=0.0)

    if session_handles and session_labels:
        fig.legend(
            handles=session_handles,
            labels=session_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=True,
            fontsize=9,
            title="Sessions (PST)",
            title_fontsize=9,
        )

    style_axes(ax)
    st.pyplot(fig)

    if show_macd:
        macd, sig, hist = compute_macd(hc)
        if not macd.dropna().empty:
            figm, axm = plt.subplots(figsize=(14, 2.6))
            axm.set_title("MACD (optional)")
            axm.plot(macd.index, macd.values, linewidth=1.4, label="MACD")
            axm.plot(sig.index, sig.values, linewidth=1.2, label="Signal")
            axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
            axm.legend(loc="lower left", framealpha=0.5, fontsize=9)
            style_axes(axm)
            st.pyplot(figm)
# =========================
# Part 7/7 — bullbear.py
# =========================
# ---------------------------
# Scanners + UI/Tabs
# ---------------------------

@st.cache_data(ttl=120)
def scan_star_daily(symbol: str, max_gap: int = 2):
    try:
        ohlc = fetch_hist_ohlc(symbol)
        if ohlc is None or ohlc.empty:
            return None
        ohlc_show = subset_by_daily_view(ohlc, daily_view)
        close = _coerce_1d_series(ohlc_show.get("Close")).dropna()
        if len(close) < 10:
            return None
        hma = compute_hma(close, period=hma_period)
        npx = compute_normalized_price(close, window=ntd_window)
        buy_mask, sell_mask = hma_npx_star_masks(close, hma, npx, max_bar_gap=max_gap)
        last_buy = buy_mask[buy_mask].index[-1] if buy_mask.any() else None
        last_sell = sell_mask[sell_mask].index[-1] if sell_mask.any() else None
        if last_buy is None and last_sell is None:
            return None
        if last_sell is None:
            t = last_buy
            side = "BLUE ★ (Buy)"
        elif last_buy is None:
            t = last_sell
            side = "PURPLE ★ (Sell)"
        else:
            t = last_buy if last_buy >= last_sell else last_sell
            side = "BLUE ★ (Buy)" if t == last_buy else "PURPLE ★ (Sell)"
        bars_since = int((len(close) - 1) - int(close.index.get_loc(t)))
        return {
            "Symbol": symbol,
            "Side": side,
            "Bars Since": bars_since,
            "Signal Time": t,
            "Signal Price": float(close.loc[t]),
            "Current Price": float(close.iloc[-1]),
        }
    except Exception:
        return None


@st.cache_data(ttl=120)
def scan_recent_npx0_up(symbol: str, max_bars: int = 2):
    try:
        ohlc = fetch_hist_ohlc(symbol)
        if ohlc is None or ohlc.empty:
            return None
        ohlc_show = subset_by_daily_view(ohlc, daily_view)
        close = _coerce_1d_series(ohlc_show.get("Close")).dropna()
        if len(close) < 20:
            return None

        x = np.arange(len(close), dtype=float)
        m, b = np.polyfit(x, close.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or m <= 0:
            return None

        npx = compute_normalized_price(close, window=ntd_window)
        cross_up0, _ = _cross_level(npx, 0.0)
        if not cross_up0.any():
            return None
        t = cross_up0[cross_up0].index[-1]
        bars_since = int((len(close) - 1) - int(close.index.get_loc(t)))
        if bars_since > int(max_bars):
            return None
        return {
            "Symbol": symbol,
            "Bars Since": bars_since,
            "Cross Time": t,
            "Current Price": float(close.iloc[-1]),
            "NPX (last)": float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan,
            "Global Slope": float(m),
        }
    except Exception:
        return None


@st.cache_data(ttl=120)
def scan_daily_sr_reversal_bbmid(symbol: str, want_side: str):
    try:
        ohlc = fetch_hist_ohlc(symbol)
        if ohlc is None or ohlc.empty:
            return None
        ohlc_show = subset_by_daily_view(ohlc, daily_view)
        close = _coerce_1d_series(ohlc_show.get("Close")).dropna()
        if len(close) < max(60, slope_lb_daily + 10):
            return None

        bb_mid, bb_up, bb_lo, _, _ = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
        yhat, up2, lo2, m, r2 = regression_with_band(close, lookback=slope_lb_daily, z=2.0)
        if not np.isfinite(r2) or r2 < float(signal_threshold):
            return None

        trig = find_slope_trigger_after_band_reversal(close, yhat, up2, lo2, horizon=rev_horizon)
        if trig is None:
            return None

        side = trig.get("side", "")
        if want_side.upper() != side.upper():
            return None

        px_now = float(close.iloc[-1])
        mid_now = float(bb_mid.dropna().iloc[-1]) if len(bb_mid.dropna()) else np.nan
        if not np.isfinite(mid_now):
            return None

        if side == "BUY":
            if not (m > 0 and px_now >= mid_now):
                return None
        else:
            if not (m < 0 and px_now <= mid_now):
                return None

        return {
            "Symbol": symbol,
            "Side": side,
            "R²": r2,
            "Slope": m,
            "Current Price": px_now,
            "BB mid": mid_now,
            "Trigger Time": trig.get("cross_time"),
            "Trigger Price": trig.get("cross_price"),
        }
    except Exception:
        return None


# ---------------------------
# ORIGINAL INTERFACE (Symbols + Run button) — restored
# ---------------------------
def original_run_controls(prefix: str, default_symbol: str):
    st.info("Pick a ticker; data is cached for ~2 minutes after first fetch. Charts stay on the last RUN ticker until you run again.")
    default_idx = 0
    try:
        default_idx = universe.index(default_symbol)
    except Exception:
        default_idx = 0

    sel = st.selectbox("Ticker:", universe, index=default_idx, key=f"{prefix}_ticker")
    view = st.radio("Chart View:", ["Daily", "Hourly", "Both"], index=0, key=f"{prefix}_view")
    hour_range = st.selectbox("Hourly lookback:", ["24h", "48h", "72h", "7d", "14d", "30d"], index=0, key=f"{prefix}_hour_range")
    run = st.button("Run Forecast", key=f"{prefix}_run")
    st.info("Click Run Forecast to display charts and forecast.")
    return sel, view, hour_range, run


# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    [
        "Original Forecast",
        "Enhanced Forecast",
        "Bull vs Bear",
        "Metrics",
        "NTD -0.75 Scanner",
        "Long-Term History",
        "Recent BUY Scanner",
        "NPX 0.5-Cross Scanner",
        "Daily Slope+BB Reversal Scanner",
        "HMA55+NPX Stars Scanner",
    ]
)

default_symbol = universe[0]
if st.session_state.last_run_params and st.session_state.last_run_params.get("mode") == mode:
    default_symbol = st.session_state.last_run_params.get("ticker", default_symbol)

# ---------------------------
# Tab 1 — Original Forecast
# ---------------------------
with tab1:
    st.header("Original Forecast")

    sel, view, hour_range, run = original_run_controls("tab1", default_symbol)

    if run:
        st.session_state.last_run_params = {"mode": mode, "ticker": sel, "view": view, "hour_range": hour_range, "ts": time.time()}

    params = st.session_state.last_run_params
    if not params or params.get("mode") != mode:
        st.warning("Click **Run Forecast** to display charts and forecast.")
    else:
        run_ticker = params["ticker"]
        run_view = params["view"]
        run_hour_range = params["hour_range"]

        if run_view in ("Daily", "Both"):
            st.subheader("Daily")
            render_daily(run_ticker, show_forecast=True)

        if run_view in ("Hourly", "Both"):
            st.subheader("Hourly")
            render_hourly(run_ticker, run_hour_range)

# ---------------------------
# Tab 2 — Enhanced Forecast
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")

    sel, view, hour_range, run = original_run_controls("tab2", default_symbol)
    if run:
        st.session_state.last_run_params = {"mode": mode, "ticker": sel, "view": view, "hour_range": hour_range, "ts": time.time()}

    params = st.session_state.last_run_params
    if not params or params.get("mode") != mode:
        st.warning("Click **Run Forecast** to display charts.")
    else:
        run_ticker = params["ticker"]
        run_view = params["view"]
        run_hour_range = params["hour_range"]

        if run_view in ("Daily", "Both"):
            st.subheader("Daily (enhanced)")
            render_daily(run_ticker, show_forecast=False)

        if run_view in ("Hourly", "Both"):
            st.subheader("Hourly (enhanced)")
            render_hourly(run_ticker, run_hour_range)

# ---------------------------
# Tab 3 — Bull vs Bear (FIXED: handle DataFrame vs Series safely)
# ---------------------------
with tab3:
    st.header("Bull vs Bear")
    st.caption("Simple bull/bear ratio from daily close changes over the selected lookback.")
    sym = st.selectbox("Symbol:", universe, key="tab3_symbol")

    s = fetch_hist_close(sym)  # guaranteed Series now
    s = _coerce_1d_series(s).dropna()

    if s.empty:
        st.warning("No data.")
    else:
        df = s.to_frame("Close").dropna()

        # Optional: recompute from period window (kept same behavior)
        df2 = yf.download(sym, period=bb_period, progress=False, auto_adjust=False)
        df2 = _normalize_yf_single_ticker_columns(df2)
        if isinstance(df2, pd.DataFrame) and (not df2.empty) and ("Close" in df2.columns):
            close2 = df2["Close"]
            if isinstance(close2, pd.DataFrame):
                close2 = close2.iloc[:, 0]
            close2 = pd.to_numeric(close2, errors="coerce").dropna()
            if not close2.empty:
                close2.index = pd.to_datetime(close2.index)
                df = close2.to_frame("Close").dropna()

        chg = df["Close"].diff()
        bull = int((chg > 0).sum())
        bear = int((chg < 0).sum())
        total = max(1, bull + bear)

        c1, c2, c3 = st.columns(3)
        c1.metric("Bull days", bull)
        c2.metric("Bear days", bear)
        c3.metric("Bull %", f"{(bull / total) * 100:.1f}%")

# ---------------------------
# Tab 4 — Metrics
# ---------------------------
with tab4:
    st.header("Metrics")
    sym = st.selectbox("Symbol:", universe, key="tab4_symbol")
    s = _coerce_1d_series(fetch_hist_close(sym)).dropna()
    s = subset_by_daily_view(s, daily_view)

    if s is None or len(s) == 0:
        st.warning("No data.")
    else:
        ret = s.pct_change().dropna()
        vol = float(ret.std() * np.sqrt(252)) if len(ret) else np.nan
        cagr = (float(s.iloc[-1]) / float(s.iloc[0])) ** (252 / max(1, len(s))) - 1 if len(s) > 10 else np.nan
        st.dataframe(
            pd.DataFrame(
                {
                    "Current": [float(s.iloc[-1])],
                    "Annualized Vol": [vol],
                    "Approx CAGR": [cagr],
                }
            ),
            use_container_width=True,
        )

# ---------------------------
# Tab 5 — NTD -0.75 Scanner (Daily)
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner (Daily)")
    st.caption("Lists symbols whose last daily NTD value is ≤ -0.75.")
    run_scan = st.button("Run NTD -0.75 Scan", key="btn_ntd075_scan")
    if run_scan:
        rows = []
        for sym in universe:
            try:
                ohlc = fetch_hist_ohlc(sym)
                if ohlc is None or ohlc.empty:
                    continue
                close = _coerce_1d_series(subset_by_daily_view(ohlc, daily_view).get("Close")).dropna()
                ntd = compute_normalized_trend(close, window=ntd_window).dropna()
                if len(ntd) and float(ntd.iloc[-1]) <= -0.75:
                    rows.append({"Symbol": sym, "NTD (last)": float(ntd.iloc[-1]), "Price": float(close.iloc[-1])})
            except Exception:
                continue
        if not rows:
            st.info("No matches.")
        else:
            st.dataframe(pd.DataFrame(rows).sort_values("NTD (last)"), use_container_width=True)

# ---------------------------
# Tab 6 — Long-Term History
# ---------------------------
with tab6:
    st.header("Long-Term History")
    sym = st.selectbox("Symbol:", universe, key="tab6_symbol")
    s = _coerce_1d_series(fetch_hist_close(sym)).dropna()
    if s.empty:
        st.warning("No data.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(s.index, s.values, label="Close")
        draw_trend_direction_line(ax, s, label_prefix="Global Trend")
        ax.set_title(f"{sym} — Max History")
        ax.legend(loc="upper left", framealpha=0.5)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# Tab 7 — Recent BUY Scanner (NPX 0-cross UP, uptrend)
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner")
    st.caption("Uptrend only: NPX crosses UP through 0.0 recently.")
    max_bars = st.slider("Max bars since NPX 0-cross UP", 0, 20, 2, 1, key="tab7_max_bars")
    run_scan = st.button("Run Recent BUY Scan", key="tab7_run")
    if run_scan:
        rows = []
        for sym in universe:
            r = scan_recent_npx0_up(sym, max_bars=max_bars)
            if r is not None:
                rows.append(r)
        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            out["Bars Since"] = out["Bars Since"].astype(int)
            out = out.sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# Tab 8 — NPX 0.5-Cross Scanner
# ---------------------------
with tab8:
    st.header("NPX 0.5-Cross Scanner (Daily)")
    st.caption("Shows symbols where NPX most recently crossed ±0.5 (either direction).")
    max_bars = st.slider("Max bars since NPX ±0.5 cross", 0, 60, 10, 1, key="tab8_max_bars")
    run_scan = st.button("Run NPX ±0.5 Scan", key="tab8_run")
    if run_scan:
        rows = []
        for sym in universe:
            try:
                ohlc = fetch_hist_ohlc(sym)
                if ohlc is None or ohlc.empty:
                    continue
                close = _coerce_1d_series(subset_by_daily_view(ohlc, daily_view).get("Close")).dropna()
                if len(close) < 20:
                    continue
                npx = compute_normalized_price(close, window=ntd_window)
                up_m05, dn_m05 = _cross_level(npx, -0.5)
                up_p05, dn_p05 = _cross_level(npx, +0.5)

                candidates = []
                if up_m05.any():
                    candidates.append(("NPX -0.5 ↑", up_m05[up_m05].index[-1]))
                if dn_m05.any():
                    candidates.append(("NPX -0.5 ↓", dn_m05[dn_m05].index[-1]))
                if up_p05.any():
                    candidates.append(("NPX +0.5 ↑", up_p05[up_p05].index[-1]))
                if dn_p05.any():
                    candidates.append(("NPX +0.5 ↓", dn_p05[dn_p05].index[-1]))
                if not candidates:
                    continue

                sig, t = sorted(candidates, key=lambda x: x[1])[-1]
                bars_since = int((len(close) - 1) - int(close.index.get_loc(t)))
                if bars_since > int(max_bars):
                    continue
                rows.append(
                    {"Symbol": sym, "Signal": sig, "Bars Since": bars_since, "Time": t, "Price": float(close.iloc[-1]), "NPX (last)": float(npx.dropna().iloc[-1])}
                )
            except Exception:
                continue
        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["Bars Since"])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# Tab 9 — Daily Slope+BB Reversal Scanner (confidence via R²)
# ---------------------------
with tab9:
    st.header("Daily Slope+BB Reversal Scanner")
    st.caption("Uses regression-fit R² as confidence. Filters: BUY = slope up + band reversal + above BB mid; SELL = slope down + band reversal + below BB mid.")
    run_scan = st.button("Run Daily Slope+BB Scan", key="tab9_run")

    if run_scan:
        buy_rows = []
        sell_rows = []
        for sym in universe:
            rb = scan_daily_sr_reversal_bbmid(sym, want_side="BUY")
            rs = scan_daily_sr_reversal_bbmid(sym, want_side="SELL")
            if rb is not None:
                buy_rows.append(rb)
            if rs is not None:
                sell_rows.append(rs)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("BUY candidates")
            if not buy_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(buy_rows).sort_values(["R²"], ascending=False)
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with c2:
            st.subheader("SELL candidates")
            if not sell_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(sell_rows).sort_values(["R²"], ascending=False)
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# Tab 10 — HMA55 + NPX Stars Scanner (Daily)
# ---------------------------
with tab10:
    st.header("HMA55+NPX Stars Scanner (Daily)")
    st.caption("BLUE ★: HMA cross UP + NPX -0.5 cross UP within ±2 bars. PURPLE ★: HMA cross DOWN + NPX +0.5 cross DOWN within ±2 bars.")
    max_bars = st.slider("Max bars since last ★", 0, 60, 10, 1, key="tab10_max_bars")
    max_gap = st.slider("Max bar gap (HMA↔NPX)", 1, 3, 2, 1, key="tab10_max_gap")
    run_scan = st.button("Run ★ Stars Scan", key="tab10_run")
    if run_scan:
        blue_rows, purple_rows = [], []
        for sym in universe:
            r = scan_star_daily(sym, max_gap=max_gap)
            if r is None:
                continue
            if int(r.get("Bars Since", 9999)) > int(max_bars):
                continue
            if str(r.get("Side", "")).startswith("BLUE"):
                blue_rows.append(r)
            else:
                purple_rows.append(r)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("BLUE ★ (Buy)")
            if not blue_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(blue_rows).sort_values(["Bars Since"])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with c2:
            st.subheader("PURPLE ★ (Sell)")
            if not purple_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(purple_rows).sort_values(["Bars Since"])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
