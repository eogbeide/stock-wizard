# -------------------------------------------------------
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

# =========================
# Styling / Page
# =========================
def _apply_mpl_theme():
    """Clean, blue-toned app-wide style to match Streamlit's default blue look."""
    try:
        plt.rcParams.update({
            "figure.facecolor": "#F7F9FC",
            "axes.facecolor":   "#FFFFFF",
            "savefig.facecolor":"#F7F9FC",
            "axes.edgecolor":   "#C8D2E3",
            "axes.labelcolor":  "#1F2937",
            "xtick.color":      "#374151",
            "ytick.color":      "#374151",
            "text.color":       "#111827",
            "grid.color":       "#D7E3F4",
            "grid.alpha":       0.28,
            "grid.linestyle":   "-",
            "axes.grid":        False,
            "legend.frameon":   True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#D0DBEC",
            "axes.titleweight": "bold",
            "axes.titlesize":   12,
            "axes.labelsize":   10,
            "font.size":        10,
        })
    except Exception:
        pass

_apply_mpl_theme()

st.set_page_config(page_title="Dashboard & Forecasts", layout="wide")
st.title("📊 Dashboard & Forecasts")

def _rerun():
    try:
        st.rerun()
    except Exception:
        pass

def auto_refresh(seconds: int = 30):
    """
    Very lightweight auto-refresh using a monotonic clock + rerun.
    Triggers a rerun once every `seconds` while the app is open.
    """
    now = time.time()
    key_last = "_auto_refresh_last"
    if key_last not in st.session_state:
        st.session_state[key_last] = now
    if now - st.session_state[key_last] >= max(5, int(seconds)):
        st.session_state[key_last] = now
        _rerun()

# =========================
# Helpers
# =========================
PACIFIC = pytz.timezone("America/Los_Angeles")

def style_axes(ax):
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

def _coerce_1d_series(obj) -> pd.Series:
    """
    Convert Series/DataFrame-like to a clean 1D float Series.
    Handles yfinance outputs that may come back as single-column DataFrames.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0]
        else:
            num = obj.select_dtypes(include=[np.number])
            if num.shape[1] >= 1:
                s = num.iloc[:, 0]
            else:
                s = obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        s = obj
    else:
        s = pd.Series(obj)

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce")
    return s

def _safe_last_float(obj) -> float:
    s = _coerce_1d_series(obj).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_pct(x, digits: int = 1) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

def fmt_price_val(x, digits: int = 4) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}f}" if np.isfinite(xv) else "n/a"

def fmt_slope(x, digits: int = 6) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}f}" if np.isfinite(xv) else "n/a"

def fmt_r2(x, digits: int = 3) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}f}" if np.isfinite(xv) else "n/a"

def label_on_left(ax, y, text, color="0.25", fontsize=8):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.002, y, f" {text} ", transform=trans, va="center", ha="left",
            fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

def label_on_right(ax, y, text, color="0.25", fontsize=8):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.998, y, f" {text} ", transform=trans, va="center", ha="right",
            fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

# =========================
# Daily-range selector helper
# =========================
def subset_by_daily_view(series_like, view_label: str) -> pd.Series:
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return s
    if isinstance(s.index, pd.DatetimeIndex):
        if s.index.tz is None:
            s.index = s.index.tz_localize(PACIFIC)
        else:
            s.index = s.index.tz_convert(PACIFIC)

    view_map = {
        "3mo":  "3mo",
        "6mo":  "6mo",
        "1y":   "1y",
        "2y":   "2y",
        "5y":   "5y",
        "10y":  "10y",
        "15y":  "15y",
        "20y":  "20y",
        "max":  None
    }
    val = view_map.get(view_label, "1y")
    if val is None:
        return s

    days_map = {
        "3mo": 92,
        "6mo": 183,
        "1y":  366,
        "2y":  366 * 2 + 1,
        "5y":  366 * 5 + 2,
        "10y": 366 * 10 + 3,
        "15y": 366 * 15 + 4,
        "20y": 366 * 20 + 5,
    }
    cutoff = pd.Timestamp.now(tz=PACIFIC) - pd.Timedelta(days=days_map[val])
    return s.loc[s.index >= cutoff]

# =========================
# Order-ticket helpers
# =========================
def pip_size_for_symbol(symbol: str, mode: str) -> float:
    if mode == "Forex":
        if "JPY" in str(symbol).upper():
            return 0.01
        return 0.0001
    return 0.01

def _diff_text(curr: float, entry: float, pip_size: float, mode: str) -> str:
    if not (np.isfinite(curr) and np.isfinite(entry)):
        return "n/a"
    diff = curr - entry
    if mode == "Forex":
        pips = diff / pip_size if pip_size and pip_size > 0 else np.nan
        return f"{diff:.5f} ({pips:+.1f} pips)"
    return f"{diff:.4f}"

def format_trade_instruction(mode: str, symbol: str, current_price: float, side: str) -> str:
    side_u = str(side).upper()
    pip = pip_size_for_symbol(symbol, mode)

    if not np.isfinite(current_price):
        return "Current price unavailable."

    if side_u == "BUY":
        entry = current_price
        tp = entry + (10 * pip if mode == "Forex" else 1.0)
        sl = entry - (5 * pip if mode == "Forex" else 0.5)
    else:
        entry = current_price
        tp = entry - (10 * pip if mode == "Forex" else 1.0)
        sl = entry + (5 * pip if mode == "Forex" else 0.5)

    return (
        f"Symbol: {symbol}\n"
        f"Side: {side_u}\n"
        f"Current Price: {entry:.5f}\n"
        f"Entry: {entry:.5f}  [{_diff_text(current_price, entry, pip, mode)} vs current]\n"
        f"Take Profit: {tp:.5f}  [{_diff_text(current_price, tp, pip, mode)} vs current]\n"
        f"Stop Loss: {sl:.5f}  [{_diff_text(current_price, sl, pip, mode)} vs current]"
    )

# =========================
# Gapless intraday helper
# =========================
def make_gapless_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a gapless OHLC frame on a continuous 5-minute grid in the
    dataframe's existing timezone. Missing bars are forward-filled from
    the previous close so plotted lines don't show large overnight/weekend gaps.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    out = df.copy().sort_index()
    tz = out.index.tz
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="5min", tz=tz)
    out = out.reindex(full_idx)

    prev_close = out["Close"].ffill()

    for col in ["Open", "High", "Low"]:
        if col not in out.columns:
            out[col] = np.nan
    if "Close" not in out.columns:
        out["Close"] = np.nan

    out["Open"]  = out["Open"].fillna(prev_close)
    out["High"]  = out["High"].fillna(prev_close)
    out["Low"]   = out["Low"].fillna(prev_close)
    out["Close"] = out["Close"].fillna(prev_close)

    return out

def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return
    n = len(real_times)
    n_ticks = int(max(2, min(n_ticks, n)))
    pos = np.linspace(0, n - 1, n_ticks, dtype=int)
    pos = np.unique(pos)
    labels = []
    for i in pos:
        try:
            labels.append(real_times[i].strftime("%m-%d %H:%M"))
        except Exception:
            labels.append(str(real_times[i]))
    ax.set_xticks(pos.tolist())
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))

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

# =========================
# Mode switching
# =========================
def _reset_run_state_for_mode_switch():
    st.session_state.run_all = False
    st.session_state.ticker = None
    for k in [
        "df_hist","df_ohlc","fc_idx","fc_vals","fc_ci",
        "intraday","chart","hour_range","mode_at_run"
    ]:
        st.session_state.pop(k, None)

if "asset_mode" not in st.session_state:
    st.session_state.asset_mode = "Forex"

st.title("📊 Dashboard & Forecasts")
mcol1, mcol2 = st.columns(2)
if mcol1.button("🌐 Forex", use_container_width=True, key="btn_mode_forex"):
    if st.session_state.asset_mode != "Forex":
        st.session_state.asset_mode = "Forex"
        _reset_run_state_for_mode_switch()
        _rerun()

if mcol2.button("🏷️ Stock", use_container_width=True, key="btn_mode_stock"):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        _rerun()

mode = st.session_state.asset_mode
st.caption(f"**Current mode:** {mode}")

# =========================
# Sidebar config
# =========================
st.sidebar.title("Configuration")
st.sidebar.markdown(f"### Asset Class: **{mode}**")

if st.sidebar.button("🧹 Clear cache"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.sidebar.success("Cleared cached data/resources.")

auto_refresh_enabled = st.sidebar.checkbox("Auto-refresh app", value=False, key="sb_auto_refresh")
if auto_refresh_enabled:
    auto_refresh_seconds = st.sidebar.slider("Auto-refresh every (sec)", 5, 300, 30, 5, key="sb_auto_refresh_sec")
    auto_refresh(auto_refresh_seconds)

if mode == "Forex":
    DEFAULT_SYMBOL = "EURUSD=X"
    universe_default = [
        "EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X",
        "EURJPY=X","GBPJPY=X","EURGBP=X","EURAUD=X","EURAZD=X" if False else "EURNZD=X",
        "AUDJPY=X","CADJPY=X","CHFJPY=X","GBPCHF=X","GBPAUD=X","AUDCAD=X","AUDCHF=X",
        "NZDJPY=X","EURCAD=X","EURCHF=X","CADCHF=X","NZDCAD=X","NZDCHF=X","GBPCAD=X"
    ]
else:
    DEFAULT_SYMBOL = "AAPL"
    universe_default = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","AVGO","NFLX","COST","QCOM",
        "INTC","MU","PLTR","ADBE","ORCL","CRM","SHOP","PYPL","UBER","JPM","BAC","XOM","CVX",
        "UNH","LLY","JNJ","PFE","KO","PEP","WMT","HD","MCD","NKE","DIS","SPY","QQQ","IWM",
        "SMH","SOXX","XLF","XLK","GLD","SLV","TLT","ARKK","COIN","SNOW","PANW","CRWD"
    ]

symbol = st.sidebar.text_input(
    f"{'Forex pair' if mode == 'Forex' else 'Stock ticker'}",
    DEFAULT_SYMBOL,
    key="sb_symbol"
).strip().upper()

daily_view = st.sidebar.selectbox(
    "Daily chart range",
    ["3mo", "6mo", "1y", "2y", "5y", "10y", "15y", "20y", "max"],
    index=2 if mode == "Forex" else 4,
    key="sb_daily_view"
)

hour_range = st.sidebar.selectbox(
    "Hourly chart window",
    ["24h", "48h", "96h"],
    index=0,
    key="sb_hour_range"
)
period_map = {"24h": "1d", "48h": "2d", "96h": "5d"}
hour_period = period_map.get(hour_range, "1d")

st.sidebar.markdown("---")
st.sidebar.subheader("Indicators / Filters")
show_hma = st.sidebar.checkbox("Show HMA", value=True, key="sb_show_hma")
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win = st.sidebar.slider("BB window", 5, 100, 20, 1, key="sb_bb_win")
bb_mult = st.sidebar.slider("BB sigma", 0.5, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema = st.sidebar.checkbox("Use EMA for BB basis", value=False, key="sb_bb_ema")

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=False, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Ichimoku conversion", 3, 30, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Ichimoku base", 5, 60, 26, 1, key="sb_ichi_base")
ichi_spanb = st.sidebar.slider("Ichimoku span B", 10, 120, 52, 1, key="sb_ichi_spanb")

show_supertrend = st.sidebar.checkbox("Show Supertrend", value=False, key="sb_show_supertrend")
atr_period = st.sidebar.slider("ATR period", 3, 50, 10, 1, key="sb_atr_period")
atr_mult = st.sidebar.slider("ATR multiplier", 0.5, 8.0, 3.0, 0.1, key="sb_atr_mult")

show_psar = st.sidebar.checkbox("Show PSAR", value=False, key="sb_show_psar")
psar_step = st.sidebar.slider("PSAR step", 0.005, 0.1, 0.02, 0.005, key="sb_psar_step")
psar_max = st.sidebar.slider("PSAR max", 0.05, 0.5, 0.2, 0.01, key="sb_psar_max")

show_ntd = st.sidebar.checkbox("Show NTD panel (Daily)", value=True, key="sb_show_ntd")
show_nrsi = st.sidebar.checkbox("Show NTD panel (Hourly)", value=True, key="sb_show_nrsi")
ntd_window = st.sidebar.slider("NTD/NPX window", 5, 200, 60, 1, key="sb_ntd_window")
shade_ntd = st.sidebar.checkbox("Shade NTD bull/bear zones", value=True, key="sb_shade_ntd")
show_npx_ntd = st.sidebar.checkbox("Overlay NPX on NTD panel", value=True, key="sb_show_npx")
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses", value=True, key="sb_mark_cross")

show_ntd_channel = st.sidebar.checkbox("Highlight price in-range on NTD panel", value=True, key="sb_show_ntd_chan")

sr_lb_daily = st.sidebar.slider("Daily S/R lookback", 10, 365, 90, 5, key="sb_sr_lb_daily")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback", 10, 480, 120, 5, key="sb_sr_lb_hourly")
sr_prox_pct = st.sidebar.slider("S/R proximity %", 0.001, 0.05, 0.01, 0.001, key="sb_sr_prox")

show_fibs = st.sidebar.checkbox("Show Fibonacci signals", value=True, key="sb_show_fibs")

slope_lb_daily  = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback (bars)", 12, 480, 120, 6, key="sb_slope_lb_hourly")

st.sidebar.subheader("MACD")
show_macd = st.sidebar.checkbox("Show MACD chart", value=False, key="sb_show_macd")

st.sidebar.subheader("Slope Reversal Probability (experimental)")
rev_hist_lb = st.sidebar.slider("History window for reversal stats (bars)", 30, 720, 240, 30, key="sb_rev_hist_lb")
rev_horizon = st.sidebar.slider("Reversal horizon (bars)", 1, 60, 10, 1, key="sb_rev_horizon")

mom_lb_daily = st.sidebar.slider("Daily ROC lookback", 2, 60, 10, 1, key="sb_mom_lb_daily")
mom_lb_hourly = st.sidebar.slider("Hourly ROC lookback", 2, 120, 10, 1, key="sb_mom_lb_hourly")

st.sidebar.markdown("---")
run_all = st.sidebar.button("▶️ Run dashboard", use_container_width=True, key="sb_run_dashboard")
if run_all:
    st.session_state.run_all = True
    st.session_state.ticker = symbol
    st.session_state.mode_at_run = mode

# =========================
# Scan universe
# =========================
st.sidebar.markdown("---")
use_default_universe = st.sidebar.checkbox("Use default scan universe", value=True, key="sb_use_default_universe")
if use_default_universe:
    universe = universe_default
else:
    universe_text = st.sidebar.text_area(
        "Custom symbols (comma/newline separated)",
        "\n".join(universe_default[:15]),
        height=180,
        key="sb_universe_text"
    )
    universe = [x.strip().upper() for x in universe_text.replace(",", "\n").splitlines() if x.strip()]

st.sidebar.caption(f"Universe size: {len(universe)}")

# =========================
# Data fetchers
# =========================
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].asfreq("D").ffill()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
        s.index = s.index.tz_localize(PACIFIC)
    elif isinstance(s.index, pd.DatetimeIndex):
        s.index = s.index.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max", progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna().asfreq("D").ffill()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
        s.index = s.index.tz_localize(PACIFIC)
    elif isinstance(s.index, pd.DatetimeIndex):
        s.index = s.index.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    need = ["Open","High","Low","Close"]
    have = [c for c in need if c in df.columns]
    if len(have) < 4:
        return pd.DataFrame()
    out = df[need].dropna(how="all")
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is None:
        out.index = out.index.tz_localize(PACIFIC)
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_convert(PACIFIC)
    return out

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    except Exception:
        pass
    try:
        df = df.tz_convert(PACIFIC)
    except Exception:
        pass
    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)
    return df

@st.cache_data(ttl=30, show_spinner=False)
def fetch_latest_yahoo_price(ticker: str, fallback: float = np.nan) -> float:
    try:
        tk = yf.Ticker(ticker)

        intraday = tk.history(period="1d", interval="1m", prepost=True, auto_adjust=False)
        if intraday is not None and not intraday.empty and "Close" in intraday.columns:
            close = pd.to_numeric(intraday["Close"], errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1])

        fast_info = getattr(tk, "fast_info", None)
        if fast_info:
            for key in ("lastPrice", "regularMarketPrice", "previousClose"):
                val = fast_info.get(key, np.nan)
                if pd.notna(val) and np.isfinite(float(val)):
                    return float(val)

        info = getattr(tk, "info", {}) or {}
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            val = info.get(key, np.nan)
            if pd.notna(val) and np.isfinite(float(val)):
                return float(val)
    except Exception:
        pass

    try:
        fallback = float(fallback)
        if np.isfinite(fallback):
            return fallback
    except Exception:
        pass
    return np.nan

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC).normalize() + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
        return idx, pd.Series(index=idx, dtype=float), pd.DataFrame(index=idx, columns=["lower","upper"])
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)

    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=30)

        fc_idx = pd.date_range(
            series.index.max() + timedelta(days=1), periods=30, freq="D",
            tz=series.index.tz if isinstance(series.index, pd.DatetimeIndex) else PACIFIC
        )
        mean = fc.predicted_mean
        if not isinstance(mean.index, pd.DatetimeIndex):
            mean.index = fc_idx
        ci = fc.conf_int()
        if not isinstance(ci.index, pd.DatetimeIndex):
            ci.index = fc_idx
        return fc_idx, mean, ci
    except Exception:
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC).normalize() + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
        return idx, pd.Series(index=idx, dtype=float), pd.DataFrame(index=idx, columns=["lower","upper"])

# =========================
# Regression / trend / oscillator helpers
# =========================
def slope_line(series_like):
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 2:
        return pd.Series(index=s.index, dtype=float), float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    return pd.Series(m * x + b, index=s.index), float(m)

def regression_with_band(series_like, lookback: int = 90, z: float = 2.0):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, float("nan"), float("nan")

    n = min(int(lookback), len(s))
    if n < 2:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty, float("nan"), float("nan")

    tail = s.iloc[-n:]
    x = np.arange(n, dtype=float)
    y = tail.to_numpy(dtype=float)

    m, b = np.polyfit(x, y, 1)
    yhat_tail = m * x + b
    resid = y - yhat_tail
    sd = float(np.nanstd(resid, ddof=1)) if len(resid) >= 2 else 0.0

    ss_res = float(np.nansum((y - yhat_tail) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    yhat = pd.Series(index=s.index, dtype=float)
    up = pd.Series(index=s.index, dtype=float)
    lo = pd.Series(index=s.index, dtype=float)

    yhat.loc[tail.index] = yhat_tail
    up.loc[tail.index] = yhat_tail + z * sd
    lo.loc[tail.index] = yhat_tail - z * sd

    return yhat, up, lo, float(m), float(r2)

def slope_reversal_probability(price_like,
                               current_slope: float,
                               hist_window: int = 240,
                               slope_window: int = 90,
                               horizon: int = 10) -> float:
    """
    Heuristic probability:
    Among past windows with slope sign similar to current_slope,
    how often did the slope flip sign within the next `horizon` bars?
    """
    s = _coerce_1d_series(price_like).dropna()
    hist_window = int(hist_window)
    slope_window = int(slope_window)
    horizon = int(horizon)

    need = max(hist_window, slope_window + horizon + 5)
    if len(s) < need or not np.isfinite(current_slope):
        return np.nan

    cur_sign = 1 if current_slope > 0 else (-1 if current_slope < 0 else 0)
    if cur_sign == 0:
        return np.nan

    start = max(slope_window, len(s) - hist_window - horizon)
    end = len(s) - horizon
    if end - start < 10:
        return np.nan

    matches = 0
    flips = 0

    vals = s.to_numpy(dtype=float)
    for i in range(start, end):
        w = vals[i - slope_window:i]
        x = np.arange(len(w), dtype=float)
        try:
            m_now, _ = np.polyfit(x, w, 1)
        except Exception:
            continue
        sign_now = 1 if m_now > 0 else (-1 if m_now < 0 else 0)
        if sign_now != cur_sign:
            continue

        fut = vals[i:i + horizon]
        x2 = np.arange(len(fut), dtype=float)
        try:
            m_future, _ = np.polyfit(x2, fut, 1)
        except Exception:
            continue
        sign_future = 1 if m_future > 0 else (-1 if m_future < 0 else 0)

        matches += 1
        if sign_future == -cur_sign:
            flips += 1

    if matches < 5:
        return np.nan
    return flips / matches

def find_band_bounce_signal(price_like, upper_like, lower_like, slope: float):
    p = _coerce_1d_series(price_like).dropna()
    u = _coerce_1d_series(upper_like).reindex(p.index)
    l = _coerce_1d_series(lower_like).reindex(p.index)
    if len(p) < 2:
        return None

    if np.isfinite(slope) and slope > 0:
        touch = p <= (l * 1.002)
        valid_idx = touch[touch].index
        if len(valid_idx):
            t = valid_idx[-1]
            return {"side": "BUY", "time": t, "price": float(p.loc[t])}
    elif np.isfinite(slope) and slope < 0:
        touch = p >= (u * 0.998)
        valid_idx = touch[touch].index
        if len(valid_idx):
            t = valid_idx[-1]
            return {"side": "SELL", "time": t, "price": float(p.loc[t])}
    return None

def _cross_series(a_like, b_like):
    a = _coerce_1d_series(a_like)
    b = _coerce_1d_series(b_like).reindex(a.index)
    d = a - b
    prev = d.shift(1)
    cross_up = (prev <= 0) & (d > 0)
    cross_dn = (prev >= 0) & (d < 0)
    return cross_up.fillna(False), cross_dn.fillna(False)

def _strict_cross_series(a_like, b_like):
    a = _coerce_1d_series(a_like)
    b = _coerce_1d_series(b_like).reindex(a.index)
    d = a - b
    prev = d.shift(1)
    cross_up = (prev < 0) & (d >= 0)
    cross_dn = (prev > 0) & (d <= 0)
    return cross_up.fillna(False), cross_dn.fillna(False)

def _bars_since_event(index_like, event_idx) -> int:
    idx = pd.Index(index_like)
    try:
        loc = int(idx.get_loc(event_idx))
        return int((len(idx) - 1) - loc)
    except Exception:
        return 10**9

def annotate_crossover(ax, x, y, side: str, note: str = ""):
    color = "tab:green" if str(side).upper() == "BUY" else "tab:red"
    marker = "^" if str(side).upper() == "BUY" else "v"
    txt = f"{side.upper()} {note}".strip()
    ax.scatter([x], [y], s=70, marker=marker, color=color, zorder=7)
    ax.annotate(
        txt, xy=(x, y), xytext=(0, 10 if marker == "^" else -14),
        textcoords="offset points", ha="center",
        va="bottom" if marker == "^" else "top",
        fontsize=8, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7)
    )

def fibonacci_levels(price_like, lookback: int = 120):
    s = _coerce_1d_series(price_like).dropna()
    if len(s) < 5:
        return {}
    w = s.iloc[-min(len(s), int(lookback)):]
    lo = float(w.min())
    hi = float(w.max())
    rng = hi - lo
    if rng <= 0:
        return {}
    levels = {
        "0.0%": lo,
        "23.6%": lo + 0.236 * rng,
        "38.2%": lo + 0.382 * rng,
        "50.0%": lo + 0.500 * rng,
        "61.8%": lo + 0.618 * rng,
        "78.6%": lo + 0.786 * rng,
        "100.0%": hi,
    }
    return levels

def fib_touch_masks(price_like, levels: dict, tolerance_pct_of_range: float = 0.015):
    p = _coerce_1d_series(price_like).dropna()
    if p.empty or not levels:
        return {}
    lo = float(min(levels.values()))
    hi = float(max(levels.values()))
    rng = hi - lo
    tol = max(rng * float(tolerance_pct_of_range), 1e-12)
    out = {}
    for name, lv in levels.items():
        out[name] = (p - float(lv)).abs() <= tol
    return out

def npx_zero_cross_masks(npx_like, level: float = 0.0):
    npx = _coerce_1d_series(npx_like).dropna()
    if len(npx) < 2:
        f = pd.Series(False, index=npx.index)
        return f, f
    prev = npx.shift(1)
    cross_up = (prev <= level) & (npx > level)
    cross_dn = (prev >= level) & (npx < level)
    return cross_up.fillna(False), cross_dn.fillna(False)

def fib_npx_zero_cross_signal_masks(price_like,
                                    npx_like,
                                    horizon_bars: int = 15,
                                    proximity_pct_of_range: float = 0.015,
                                    npx_level: float = 0.0):
    p = _coerce_1d_series(price_like).dropna()
    npx = _coerce_1d_series(npx_like).reindex(p.index)
    if len(p) < 10:
        f = pd.Series(False, index=p.index)
        return f, f, {}

    fibs = fibonacci_levels(p, lookback=min(180, len(p)))
    touches = fib_touch_masks(p, fibs, tolerance_pct_of_range=proximity_pct_of_range)
    if not touches:
        f = pd.Series(False, index=p.index)
        return f, f, fibs

    touch_any = pd.Series(False, index=p.index)
    for _, mask in touches.items():
        touch_any = touch_any | mask.reindex(p.index, fill_value=False)

    up0, dn0 = npx_zero_cross_masks(npx, level=npx_level)

    buy_mask = pd.Series(False, index=p.index)
    sell_mask = pd.Series(False, index=p.index)

    hz = max(1, int(horizon_bars))

    touch_idx = np.where(touch_any.fillna(False).to_numpy())[0]
    up_idx = np.where(up0.reindex(p.index, fill_value=False).to_numpy())[0]
    dn_idx = np.where(dn0.reindex(p.index, fill_value=False).to_numpy())[0]

    for i in touch_idx:
        if np.any((up_idx >= i) & (up_idx <= i + hz)):
            buy_mask.iloc[i] = True
        if np.any((dn_idx >= i) & (dn_idx <= i + hz)):
            sell_mask.iloc[i] = True

    return buy_mask, sell_mask, fibs

def overlay_fib_npx_signals(ax, price_like, fib_buy_mask, fib_sell_mask):
    p = _coerce_1d_series(price_like).dropna()
    if p.empty:
        return
    fb = fib_buy_mask.reindex(p.index, fill_value=False) if fib_buy_mask is not None else pd.Series(False, index=p.index)
    fs = fib_sell_mask.reindex(p.index, fill_value=False) if fib_sell_mask is not None else pd.Series(False, index=p.index)

    if fb.any():
        ax.scatter(
            p.index[fb], p.loc[fb],
            marker="^", s=55, color="tab:green", zorder=7, label="Fib+NPX Buy"
        )
    if fs.any():
        ax.scatter(
            p.index[fs], p.loc[fs],
            marker="v", s=55, color="tab:red", zorder=7, label="Fib+NPX Sell"
        )

def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.pct_change(int(n))

def compute_rsi(series_like, period: int = 14) -> pd.Series:
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / \
         dn.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series_like, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        e = pd.Series(dtype=float)
        return e, e, e
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def _wma(series_like, n: int):
    s = _coerce_1d_series(series_like)
    w = np.arange(1, int(n) + 1, dtype=float)
    return s.rolling(int(n)).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def compute_hma(series_like, period: int = 55):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    n = int(period)
    if n < 2:
        return s.copy()
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    h = 2 * _wma(s, half) - _wma(s, n)
    return _wma(h, sqrt_n)

def compute_bbands(series_like, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        e = pd.Series(dtype=float)
        return e, e, e, e, e

    w = int(window)
    if use_ema:
        mid = s.ewm(span=w, adjust=False).mean()
        sd = s.ewm(span=w, adjust=False).std(bias=False)
    else:
        mid = s.rolling(w, min_periods=w).mean()
        sd = s.rolling(w, min_periods=w).std(ddof=1)

    up = mid + float(mult) * sd
    lo = mid - float(mult) * sd

    pctb = (s - lo) / (up - lo)
    nbb = (s - mid) / (float(mult) * sd)
    return mid, up, lo, pctb, nbb

def compute_normalized_trend(series_like, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 3:
        return pd.Series(index=s.index, dtype=float)

    w = int(window)
    out = pd.Series(index=s.index, dtype=float)

    vals = s.to_numpy(dtype=float)
    for i in range(w - 1, len(s)):
        seg = vals[i - w + 1:i + 1]
        x = np.arange(w, dtype=float)
        try:
            m, _ = np.polyfit(x, seg, 1)
            denom = np.nanstd(seg, ddof=1)
            out.iloc[i] = (m * w) / denom if denom and np.isfinite(denom) and denom > 0 else np.nan
        except Exception:
            out.iloc[i] = np.nan

    out = out / 3.0
    return out.clip(-1.0, 1.0)

def compute_normalized_price(series_like, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return pd.Series(dtype=float)

    mid = s.rolling(int(window), min_periods=int(window)).mean()
    sd = s.rolling(int(window), min_periods=int(window)).std(ddof=1)
    z = (s - mid) / sd
    npx = z / 2.0
    return npx.clip(-1.0, 1.0)

def shade_ntd_regions(ax, ntd_like):
    ntd = _coerce_1d_series(ntd_like).dropna()
    if ntd.empty:
        return
    x = ntd.index
    y = ntd.values
    ax.fill_between(x, 0, y, where=(y >= 0), alpha=0.15, interpolate=True)
    ax.fill_between(x, 0, y, where=(y < 0), alpha=0.10, interpolate=True)

def draw_trend_direction_line(ax, trend_like, label: str = "Trend Direction"):
    t = _coerce_1d_series(trend_like).dropna()
    if t.empty:
        return
    ax.plot(t.index, np.sign(t.values), linewidth=1.0, linestyle="--", label=label)

def overlay_npx_on_ntd(ax, npx_like, ntd_like, mark_crosses: bool = True):
    npx = _coerce_1d_series(npx_like).dropna()
    ntd = _coerce_1d_series(ntd_like).reindex(npx.index)

    ax.plot(npx.index, npx.values, label="NPX", linewidth=1.0, alpha=0.95)

    if mark_crosses:
        up_mask, dn_mask = _strict_cross_series(npx, ntd)
        if up_mask.any():
            ax.scatter(npx.index[up_mask], npx.loc[up_mask], marker="^", s=28, label="NPX↑NTD")
        if dn_mask.any():
            ax.scatter(npx.index[dn_mask], npx.loc[dn_mask], marker="v", s=28, label="NPX↓NTD")

def ichimoku_lines(high_like, low_like, close_like, conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = False):
    h = _coerce_1d_series(high_like)
    l = _coerce_1d_series(low_like).reindex(h.index)
    c = _coerce_1d_series(close_like).reindex(h.index)

    tenkan = (h.rolling(conv).max() + l.rolling(conv).min()) / 2
    kijun = (h.rolling(base).max() + l.rolling(base).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_bv = (h.rolling(span_b).max() + l.rolling(span_b).min()) / 2
    chikou = c.shift(-base)

    if shift_cloud:
        span_a = span_a.shift(base)
        span_bv = span_bv.shift(base)

    return tenkan, kijun, span_a, span_bv, chikou

def _compute_atr_from_ohlc(ohlc: pd.DataFrame, period: int = 10) -> pd.Series:
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.Series(dtype=float)
    high = _coerce_1d_series(ohlc["High"])
    low = _coerce_1d_series(ohlc["Low"]).reindex(high.index)
    close = _coerce_1d_series(ohlc["Close"]).reindex(high.index)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / int(period), adjust=False, min_periods=int(period)).mean()

def compute_supertrend(ohlc: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.DataFrame()

    df = ohlc.copy()
    high = _coerce_1d_series(df["High"])
    low = _coerce_1d_series(df["Low"]).reindex(high.index)
    close = _coerce_1d_series(df["Close"]).reindex(high.index)

    atr = _compute_atr_from_ohlc(df, period=int(atr_period))
    hl2 = (high + low) / 2.0

    upper_basic = hl2 + float(atr_mult) * atr
    lower_basic = hl2 - float(atr_mult) * atr

    upper_final = upper_basic.copy()
    lower_final = lower_basic.copy()

    for i in range(1, len(df)):
        idx = df.index[i]
        prev = df.index[i - 1]

        if np.isfinite(upper_basic.loc[idx]) and np.isfinite(upper_final.loc[prev]) and np.isfinite(close.loc[prev]):
            if (upper_basic.loc[idx] < upper_final.loc[prev]) or (close.loc[prev] > upper_final.loc[prev]):
                upper_final.loc[idx] = upper_basic.loc[idx]
            else:
                upper_final.loc[idx] = upper_final.loc[prev]

        if np.isfinite(lower_basic.loc[idx]) and np.isfinite(lower_final.loc[prev]) and np.isfinite(close.loc[prev]):
            if (lower_basic.loc[idx] > lower_final.loc[prev]) or (close.loc[prev] < lower_final.loc[prev]):
                lower_final.loc[idx] = lower_basic.loc[idx]
            else:
                lower_final.loc[idx] = lower_final.loc[prev]

    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    if len(df):
        st_line.iloc[0] = upper_final.iloc[0]
        direction.iloc[0] = -1

    for i in range(1, len(df)):
        idx = df.index[i]
        prev = df.index[i - 1]

        if close.loc[idx] > upper_final.loc[prev]:
            direction.loc[idx] = 1
        elif close.loc[idx] < lower_final.loc[prev]:
            direction.loc[idx] = -1
        else:
            direction.loc[idx] = direction.loc[prev]

        st_line.loc[idx] = lower_final.loc[idx] if direction.loc[idx] > 0 else upper_final.loc[idx]

    return pd.DataFrame({"ST": st_line, "DIR": direction, "ATR": atr}, index=df.index)

def compute_psar_from_ohlc(ohlc: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.DataFrame()

    high = _coerce_1d_series(ohlc["High"]).reset_index(drop=True)
    low = _coerce_1d_series(ohlc["Low"]).reset_index(drop=True)
    close = _coerce_1d_series(ohlc["Close"]).reset_index(drop=True)

    n = len(ohlc)
    psar = np.full(n, np.nan)
    bull = np.full(n, True, dtype=bool)
    af = step
    ep = high.iloc[0]
    psar[0] = low.iloc[0]

    for i in range(1, n):
        prev_psar = psar[i - 1]

        if bull[i - 1]:
            psar[i] = prev_psar + af * (ep - prev_psar)
            if i >= 2:
                psar[i] = min(psar[i], low.iloc[i - 1], low.iloc[i - 2])
            else:
                psar[i] = min(psar[i], low.iloc[i - 1])

            if low.iloc[i] < psar[i]:
                bull[i] = False
                psar[i] = ep
                ep = low.iloc[i]
                af = step
            else:
                bull[i] = True
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            if i >= 2:
                psar[i] = max(psar[i], high.iloc[i - 1], high.iloc[i - 2])
            else:
                psar[i] = max(psar[i], high.iloc[i - 1])

            if high.iloc[i] > psar[i]:
                bull[i] = True
                psar[i] = ep
                ep = high.iloc[i]
                af = step
            else:
                bull[i] = False
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)

    out = pd.DataFrame(
        {
            "PSAR": psar,
            "BULL": bull.astype(int),
        },
        index=ohlc.index
    )
    return out

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
# Forex sessions + Yahoo news
# =========================
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        return opens, closes
    loc = idx.tz_convert(session_tz)
    dates = pd.Index(loc.date).unique()
    for d in dates:
        try:
            o_local = pd.Timestamp(datetime(d.year, d.month, d.day, open_hr, 0), tz=session_tz)
            c_local = pd.Timestamp(datetime(d.year, d.month, d.day, close_hr, 0), tz=session_tz)
            o = o_local.tz_convert(idx.tz)
            c = c_local.tz_convert(idx.tz)
            if idx.min() <= o <= idx.max():
                opens.append(o)
            if idx.min() <= c <= idx.max():
                closes.append(c)
        except Exception:
            pass
    return opens, closes

def compute_session_lines(real_times: pd.DatetimeIndex):
    london_open, london_close = session_markers_for_index(real_times, LDN_TZ, 8, 17)
    ny_open, ny_close = session_markers_for_index(real_times, NY_TZ, 9, 16)
    return {
        "London Open": london_open,
        "London Close": london_close,
        "NY Open": ny_open,
        "NY Close": ny_close,
    }

def draw_session_lines(ax, sessions: dict, alpha: float = 0.28):
    style_map = {
        "London Open":  dict(ls="--", lw=1.0),
        "London Close": dict(ls=":",  lw=1.0),
        "NY Open":      dict(ls="--", lw=1.0),
        "NY Close":     dict(ls=":",  lw=1.0),
    }
    handles = []
    labels = []
    for label, times in sessions.items():
        first = True
        for x in times:
            ax.axvline(x, alpha=alpha, **style_map.get(label, {}))
            if first:
                handles.append(Line2D([0], [0], **style_map.get(label, {})))
                labels.append(label)
                first = False
    return handles, labels

@st.cache_data(ttl=120, show_spinner=False)
def fetch_yf_news(symbol: str, window_days: int = 7) -> pd.DataFrame:
    rows = []
    try:
        news_list = yf.Ticker(symbol).news or []
    except Exception:
        news_list = []
    for item in news_list:
        ts = item.get("providerPublishTime") or item.get("pubDate")
        title = item.get("title", "")
        link = item.get("link", "")
        if ts is None:
            continue
        try:
            if isinstance(ts, (int, float, np.integer, np.floating)):
                t = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(PACIFIC)
            else:
                t = pd.to_datetime(ts, utc=True).tz_convert(PACIFIC)
        except Exception:
            continue
        rows.append({"time": t, "title": title, "link": link})
    if not rows:
        return pd.DataFrame(columns=["time", "title", "link"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["time", "title"]).sort_values("time")
    cutoff = pd.Timestamp.now(tz=PACIFIC) - pd.Timedelta(days=int(window_days))
    return df[df["time"] >= cutoff].reset_index(drop=True)

def draw_news_markers(ax, times_list, label: str = "News", alpha: float = 0.18):
    if not times_list:
        return
    first = True
    for t in times_list:
        ax.axvline(t, linestyle="-.", linewidth=0.9, alpha=alpha, label=label if first else None)
        first = False

# =========================
# Channel state overlay for NTD panel
# =========================
def channel_state_series(price_like, support_like, resistance_like):
    """
    State:
      -1 => below support
       0 => within [support, resistance]
      +1 => above resistance
    """
    p = _coerce_1d_series(price_like).dropna()
    s = _coerce_1d_series(support_like).reindex(p.index)
    r = _coerce_1d_series(resistance_like).reindex(p.index)

    state = pd.Series(index=p.index, dtype=float)
    state[p < s] = -1.0
    state[(p >= s) & (p <= r)] = 0.0
    state[p > r] = 1.0
    return state

def _true_spans(mask_like):
    mask = pd.Series(mask_like).fillna(False).astype(bool)
    spans = []
    in_run = False
    start = None
    for i, v in enumerate(mask.values):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            spans.append((start, i - 1))
            in_run = False
    if in_run:
        spans.append((start, len(mask) - 1))
    return spans

def overlay_inrange_on_ntd(ax, price_like, support_like, resistance_like):
    p = _coerce_1d_series(price_like).dropna()
    if p.empty:
        return

    state = channel_state_series(p, support_like, resistance_like).reindex(p.index)
    in_range = (state == 0).fillna(False)
    above_r = (state > 0).fillna(False)
    below_s = (state < 0).fillna(False)

    for a, b in _true_spans(in_range):
        ax.axvspan(p.index[a], p.index[b], alpha=0.08, linewidth=0)

    enter_from_below = (below_s.shift(1, fill_value=False) & in_range)
    enter_from_above = (above_r.shift(1, fill_value=False) & in_range)

    if enter_from_below.any():
        ax.scatter(p.index[enter_from_below], np.zeros(enter_from_below.sum()),
                   marker="^", s=60, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(p.index[enter_from_above], np.zeros(enter_from_above.sum()),
                   marker="v", s=60, color="tab:orange", zorder=7, label="Enter from R")

    last = state.dropna().iloc[-1] if state.dropna().shape[0] else np.nan
    if np.isfinite(last):
        state_txt = "Above R" if last > 0 else ("Below S" if last < 0 else "Inside Channel")
        ax.text(
            0.995, 0.98, state_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.75)
        )

# =========================
# S/R helpers
# =========================
def support_resistance(price_like, lookback: int = 90):
    p = _coerce_1d_series(price_like).dropna()
    if p.empty:
        return float("nan"), float("nan")
    window = p.iloc[-min(len(p), int(lookback)):]
    return float(window.min()), float(window.max())

def proximity_to_level(price: float, level: float) -> float:
    if not (np.isfinite(price) and np.isfinite(level) and level != 0):
        return np.nan
    return abs(price - level) / abs(level)

# =========================
# Forecast Signal helpers
# =========================
def ntd_direction_signal(ntd_like):
    ntd = _coerce_1d_series(ntd_like).dropna()
    if len(ntd) < 2:
        return "Neutral"
    last = float(ntd.iloc[-1])
    prev = float(ntd.iloc[-2])
    if last > 0 and last >= prev:
        return "Buy"
    if last < 0 and last <= prev:
        return "Sell"
    return "Neutral"

def ntd_reversal_signal(ntd_like):
    ntd = _coerce_1d_series(ntd_like).dropna()
    if len(ntd) < 3:
        return "Neutral"
    a, b, c = map(float, ntd.iloc[-3:])
    if a < b and c < b and b > 0:
        return "Sell"
    if a > b and c > b and b < 0:
        return "Buy"
    return "Neutral"

def ntd_zero_cross_signal(ntd_like):
    ntd = _coerce_1d_series(ntd_like).dropna()
    if len(ntd) < 2:
        return "Neutral"
    prev, last = float(ntd.iloc[-2]), float(ntd.iloc[-1])
    if prev <= 0 < last:
        return "Buy"
    if prev >= 0 > last:
        return "Sell"
    return "Neutral"

def npx_cross_ntd_signal(npx_like, ntd_like):
    up, dn = _strict_cross_series(npx_like, ntd_like)
    if up.any() and up.iloc[-1]:
        return "Buy"
    if dn.any() and dn.iloc[-1]:
        return "Sell"
    return "Neutral"

def npx_zero_cross_signal(npx_like):
    npx = _coerce_1d_series(npx_like).dropna()
    if len(npx) < 2:
        return "Neutral"
    prev, last = float(npx.iloc[-2]), float(npx.iloc[-1])
    if prev <= 0 < last:
        return "Buy"
    if prev >= 0 > last:
        return "Sell"
    return "Neutral"

def slope_direction_signal(slope: float):
    if not np.isfinite(slope):
        return "Neutral"
    return "Buy" if slope > 0 else ("Sell" if slope < 0 else "Neutral")

def slope_reversal_trigger(price_like, lookback: int = 90):
    s = _coerce_1d_series(price_like).dropna()
    if len(s) < max(lookback, 20):
        return "Neutral"
    w = s.iloc[-int(lookback):]
    x = np.arange(len(w), dtype=float)
    y = w.to_numpy(dtype=float)
    try:
        m, b = np.polyfit(x, y, 1)
        fit = m * x + b
        resid = y - fit
        z = (resid[-1] - np.nanmean(resid)) / (np.nanstd(resid, ddof=1) if len(resid) > 2 else np.nan)
    except Exception:
        return "Neutral"

    if np.isfinite(m) and m > 0 and np.isfinite(z) and z > 1.5:
        return "Sell"
    if np.isfinite(m) and m < 0 and np.isfinite(z) and z < -1.5:
        return "Buy"
    return "Neutral"

def support_resistance_signal(price_like, lookback: int = 90, prox_pct: float = 0.01):
    s = _coerce_1d_series(price_like).dropna()
    if s.empty:
        return "Neutral"
    curr = float(s.iloc[-1])
    sup, res = support_resistance(s, lookback=lookback)
    if proximity_to_level(curr, sup) <= prox_pct:
        return "Buy"
    if proximity_to_level(curr, res) <= prox_pct:
        return "Sell"
    return "Neutral"

def fib_signal(price_like, lookback: int = 120, prox_pct_of_range: float = 0.015):
    p = _coerce_1d_series(price_like).dropna()
    if p.empty:
        return "Neutral"
    fibs = fibonacci_levels(p, lookback=lookback)
    if not fibs:
        return "Neutral"

    lo = float(min(fibs.values()))
    hi = float(max(fibs.values()))
    rng = hi - lo
    tol = max(rng * float(prox_pct_of_range), 1e-12)
    curr = float(p.iloc[-1])

    nearest_name, nearest_val = min(fibs.items(), key=lambda kv: abs(curr - kv[1]))

    bullish_levels = {"23.6%", "38.2%", "50.0%", "61.8%"}
    bearish_levels = {"78.6%", "100.0%"}

    if abs(curr - nearest_val) <= tol:
        if nearest_name in bullish_levels:
            return "Buy"
        if nearest_name in bearish_levels:
            return "Sell"
    return "Neutral"

# =========================
# Run / cache state
# =========================
if st.session_state.get("run_all", False):
    symbol = st.session_state.get("ticker", symbol)
else:
    st.info("Set your configuration in the sidebar, then click **Run dashboard**.")
    st.stop()

if st.session_state.get("mode_at_run") != mode:
    st.warning("Asset mode changed since last run. Please click **Run dashboard** again.")
    st.stop()

# Daily series / OHLC
if daily_view == "max":
    hist = fetch_hist_max(symbol)
else:
    hist = fetch_hist(symbol)

ohlc_daily = fetch_hist_ohlc(symbol)

if hist.empty:
    st.error(f"No daily data found for {symbol}.")
    st.stop()

hist_view = subset_by_daily_view(hist, daily_view)
if hist_view.empty:
    hist_view = hist.copy()

# Intraday
intraday = fetch_intraday(symbol, period=hour_period)

# Latest Yahoo current price check
daily_last_close = _safe_last_float(hist_view)
intraday_last_close = (
    _safe_last_float(intraday["Close"])
    if (intraday is not None and not intraday.empty and "Close" in intraday.columns)
    else float("nan")
)
current_price = fetch_latest_yahoo_price(
    symbol,
    fallback=intraday_last_close if np.isfinite(intraday_last_close) else daily_last_close
)

if not np.isfinite(current_price):
    current_price = intraday_last_close if np.isfinite(intraday_last_close) else daily_last_close

# =========================
# Derived indicators
# =========================
slope_daily_line, slope_daily_val = slope_line(hist_view)
reg_d, up_d, lo_d, reg_slope_d, reg_r2_d = regression_with_band(hist_view, lookback=slope_lb_daily, z=2.0)

ntd_daily = compute_normalized_trend(hist_view, window=ntd_window)
npx_daily = compute_normalized_price(hist_view, window=ntd_window)
roc_daily = compute_roc(hist_view, n=mom_lb_daily)
rsi_daily = compute_rsi(hist_view, period=14)
macd_daily, macd_sig_daily, macd_hist_daily = compute_macd(hist_view)

hma_daily = compute_hma(hist_view, period=hma_period) if show_hma else pd.Series(dtype=float)
bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = (
    compute_bbands(hist_view, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
    if show_bbands else
    (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
)

tenkan_d = kijun_d = span_a_d = span_b_d = chikou_d = pd.Series(dtype=float)
if show_ichi and not ohlc_daily.empty:
    tenkan_d, kijun_d, span_a_d, span_b_d, chikou_d = ichimoku_lines(
        ohlc_daily["High"].reindex(hist_view.index).ffill(),
        ohlc_daily["Low"].reindex(hist_view.index).ffill(),
        hist_view,
        conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
    )

supertrend_d = pd.DataFrame()
if show_supertrend and not ohlc_daily.empty:
    supertrend_d = compute_supertrend(ohlc_daily.reindex(hist_view.index).dropna(), atr_period=atr_period, atr_mult=atr_mult)

psar_d = pd.DataFrame()
if show_psar and not ohlc_daily.empty:
    psar_d = compute_psar_from_ohlc(ohlc_daily.reindex(hist_view.index).dropna(), step=psar_step, max_step=psar_max)

pivots_d = current_daily_pivots(ohlc_daily)

fib_buy_d = fib_sell_d = pd.Series(False, index=hist_view.index)
fibs_d = {}
if show_fibs:
    fib_buy_d, fib_sell_d, fibs_d = fib_npx_zero_cross_signal_masks(
        hist_view, npx_daily, horizon_bars=15, proximity_pct_of_range=0.015, npx_level=0.0
    )

sup_d, res_d = support_resistance(hist_view, lookback=sr_lb_daily)
rev_prob_d = slope_reversal_probability(
    hist_view, reg_slope_d, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon
)

# Hourly derived
if intraday is not None and not intraday.empty and "Close" in intraday.columns:
    intraday_close = _coerce_1d_series(intraday["Close"]).dropna()
else:
    intraday_close = pd.Series(dtype=float)

reg_h, up_h, lo_h, reg_slope_h, reg_r2_h = regression_with_band(intraday_close, lookback=slope_lb_hourly, z=2.0)
ntd_hour = compute_normalized_trend(intraday_close, window=min(ntd_window, max(10, len(intraday_close)))) if not intraday_close.empty else pd.Series(dtype=float)
npx_hour = compute_normalized_price(intraday_close, window=min(ntd_window, max(10, len(intraday_close)))) if not intraday_close.empty else pd.Series(dtype=float)
roc_hour = compute_roc(intraday_close, n=mom_lb_hourly) if not intraday_close.empty else pd.Series(dtype=float)
rsi_hour = compute_rsi(intraday_close, period=14) if not intraday_close.empty else pd.Series(dtype=float)
macd_hour, macd_sig_hour, macd_hist_hour = compute_macd(intraday_close) if not intraday_close.empty else (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))

hma_hour = compute_hma(intraday_close, period=min(hma_period, max(5, len(intraday_close)))) if (show_hma and not intraday_close.empty) else pd.Series(dtype=float)
bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = (
    compute_bbands(intraday_close, window=min(bb_win, max(5, len(intraday_close))), mult=bb_mult, use_ema=bb_use_ema)
    if (show_bbands and not intraday_close.empty) else
    (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
)

tenkan_h = kijun_h = span_a_h = span_b_h = chikou_h = pd.Series(dtype=float)
if show_ichi and intraday is not None and not intraday.empty and {"High","Low","Close"}.issubset(intraday.columns):
    tenkan_h, kijun_h, span_a_h, span_b_h, chikou_h = ichimoku_lines(
        intraday["High"], intraday["Low"], intraday["Close"],
        conv=max(3, min(ichi_conv, max(5, len(intraday_close)//3 if len(intraday_close) else ichi_conv))),
        base=max(5, min(ichi_base, max(8, len(intraday_close)//2 if len(intraday_close) else ichi_base))),
        span_b=max(10, min(ichi_spanb, max(12, len(intraday_close)))),
        shift_cloud=False
    )

supertrend_h = pd.DataFrame()
if show_supertrend and intraday is not None and not intraday.empty and {"High","Low","Close"}.issubset(intraday.columns):
    supertrend_h = compute_supertrend(intraday[["High","Low","Close"]], atr_period=atr_period, atr_mult=atr_mult)

psar_h = pd.DataFrame()
if show_psar and intraday is not None and not intraday.empty and {"High","Low","Close"}.issubset(intraday.columns):
    psar_h = compute_psar_from_ohlc(intraday[["High","Low","Close"]], step=psar_step, max_step=psar_max)

fib_buy_h = fib_sell_h = pd.Series(False, index=intraday_close.index)
fibs_h = {}
if show_fibs and not intraday_close.empty:
    fib_buy_h, fib_sell_h, fibs_h = fib_npx_zero_cross_signal_masks(
        intraday_close, npx_hour, horizon_bars=12, proximity_pct_of_range=0.015, npx_level=0.0
    )

sup_h, res_h = support_resistance(intraday_close, lookback=sr_lb_hourly) if not intraday_close.empty else (np.nan, np.nan)
rev_prob_h = slope_reversal_probability(
    intraday_close, reg_slope_h, hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon
) if not intraday_close.empty else np.nan

# Forecast
fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(hist_view)

# =========================
# Header metrics
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Symbol", symbol)
m2.metric("Current Price", fmt_price_val(current_price, digits=5 if mode == "Forex" else 4))
m3.metric("Daily Trend Slope", fmt_slope(reg_slope_d))
m4.metric("Hourly Trend Slope", fmt_slope(reg_slope_h))

# =========================
# Signal dashboard
# =========================
st.subheader("Signal Dashboard")

sig_rows = []

daily_dir_sig = ntd_direction_signal(ntd_daily)
daily_rev_sig = ntd_reversal_signal(ntd_daily)
daily_zero_sig = ntd_zero_cross_signal(ntd_daily)
daily_npx_ntd_sig = npx_cross_ntd_signal(npx_daily, ntd_daily)
daily_npx_zero_sig = npx_zero_cross_signal(npx_daily)
daily_slope_sig = slope_direction_signal(reg_slope_d)
daily_slope_rev_sig = slope_reversal_trigger(hist_view, lookback=slope_lb_daily)
daily_sr_sig = support_resistance_signal(hist_view, lookback=sr_lb_daily, prox_pct=sr_prox_pct)
daily_fib_sig = fib_signal(hist_view, lookback=min(180, len(hist_view)), prox_pct_of_range=0.015)

sig_rows.extend([
    {"Scope": "Daily", "Signal": "NTD Direction", "Value": daily_dir_sig},
    {"Scope": "Daily", "Signal": "NTD Reversal", "Value": daily_rev_sig},
    {"Scope": "Daily", "Signal": "NTD Zero Cross", "Value": daily_zero_sig},
    {"Scope": "Daily", "Signal": "NPX vs NTD", "Value": daily_npx_ntd_sig},
    {"Scope": "Daily", "Signal": "NPX Zero Cross", "Value": daily_npx_zero_sig},
    {"Scope": "Daily", "Signal": "Slope Direction", "Value": daily_slope_sig},
    {"Scope": "Daily", "Signal": "Slope Reversal", "Value": daily_slope_rev_sig},
    {"Scope": "Daily", "Signal": "Support/Resistance", "Value": daily_sr_sig},
    {"Scope": "Daily", "Signal": "Fibonacci", "Value": daily_fib_sig},
])

if not intraday_close.empty:
    hour_dir_sig = ntd_direction_signal(ntd_hour)
    hour_rev_sig = ntd_reversal_signal(ntd_hour)
    hour_zero_sig = ntd_zero_cross_signal(ntd_hour)
    hour_npx_ntd_sig = npx_cross_ntd_signal(npx_hour, ntd_hour)
    hour_npx_zero_sig = npx_zero_cross_signal(npx_hour)
    hour_slope_sig = slope_direction_signal(reg_slope_h)
    hour_slope_rev_sig = slope_reversal_trigger(intraday_close, lookback=min(slope_lb_hourly, max(20, len(intraday_close))))
    hour_sr_sig = support_resistance_signal(intraday_close, lookback=min(sr_lb_hourly, max(10, len(intraday_close))), prox_pct=sr_prox_pct)
    hour_fib_sig = fib_signal(intraday_close, lookback=min(180, len(intraday_close)), prox_pct_of_range=0.015)

    sig_rows.extend([
        {"Scope": "Hourly", "Signal": "NTD Direction", "Value": hour_dir_sig},
        {"Scope": "Hourly", "Signal": "NTD Reversal", "Value": hour_rev_sig},
        {"Scope": "Hourly", "Signal": "NTD Zero Cross", "Value": hour_zero_sig},
        {"Scope": "Hourly", "Signal": "NPX vs NTD", "Value": hour_npx_ntd_sig},
        {"Scope": "Hourly", "Signal": "NPX Zero Cross", "Value": hour_npx_zero_sig},
        {"Scope": "Hourly", "Signal": "Slope Direction", "Value": hour_slope_sig},
        {"Scope": "Hourly", "Signal": "Slope Reversal", "Value": hour_slope_rev_sig},
        {"Scope": "Hourly", "Signal": "Support/Resistance", "Value": hour_sr_sig},
        {"Scope": "Hourly", "Signal": "Fibonacci", "Value": hour_fib_sig},
    ])

sig_df = pd.DataFrame(sig_rows)
st.dataframe(sig_df, use_container_width=True, hide_index=True)

# =========================
# Trade ticket
# =========================
st.subheader("Trade Ticket")
ticket_side = st.selectbox("Side", ["BUY", "SELL"], index=0)
ticket_text = format_trade_instruction(mode, symbol, current_price, ticket_side)
st.code(ticket_text)

# =========================
# Forecast plot
# =========================
st.subheader("30-Day Forecast")
fig_fc, ax_fc = plt.subplots(figsize=(12, 5))
ax_fc.plot(hist_view.index, hist_view.values, label="History", linewidth=1.5)
if fc_vals is not None and len(fc_vals):
    ax_fc.plot(fc_vals.index, fc_vals.values, label="Forecast", linewidth=1.5)
if fc_ci is not None and not fc_ci.empty and fc_ci.shape[1] >= 2:
    lo_ci = pd.to_numeric(fc_ci.iloc[:, 0], errors="coerce")
    hi_ci = pd.to_numeric(fc_ci.iloc[:, 1], errors="coerce")
    ax_fc.fill_between(fc_ci.index, lo_ci.values, hi_ci.values, alpha=0.15, label="Forecast CI")
ax_fc.set_title(f"{symbol} Daily History + 30-Day Forecast")
ax_fc.legend(loc="best")
style_axes(ax_fc)
st.pyplot(fig_fc, use_container_width=True)

# =========================
# Daily chart
# =========================
st.subheader("Daily Price Chart")
fig_d, ax_d = plt.subplots(figsize=(14, 6))

ax_d.plot(hist_view.index, hist_view.values, label="Close", linewidth=1.4)
if not reg_d.dropna().empty:
    ax_d.plot(reg_d.index, reg_d.values, label="Regression", linewidth=1.2, linestyle="--")
if not up_d.dropna().empty and not lo_d.dropna().empty:
    ax_d.plot(up_d.index, up_d.values, linewidth=1.0, linestyle=":")
    ax_d.plot(lo_d.index, lo_d.values, linewidth=1.0, linestyle=":")
if show_hma and not hma_daily.dropna().empty:
    ax_d.plot(hma_daily.index, hma_daily.values, label=f"HMA({hma_period})", linewidth=1.1)
if show_bbands and not bb_mid_d.dropna().empty:
    ax_d.plot(bb_mid_d.index, bb_mid_d.values, label="BB Mid", linewidth=1.0, alpha=0.9)
    ax_d.plot(bb_up_d.index, bb_up_d.values, linewidth=0.9, linestyle=":")
    ax_d.plot(bb_lo_d.index, bb_lo_d.values, linewidth=0.9, linestyle=":")
if show_ichi and not kijun_d.dropna().empty:
    ax_d.plot(kijun_d.index, kijun_d.values, label="Kijun", linewidth=1.0)
if show_supertrend and not supertrend_d.empty and "ST" in supertrend_d.columns:
    ax_d.plot(supertrend_d.index, supertrend_d["ST"].values, label="Supertrend", linewidth=1.1)
if show_psar and not psar_d.empty and "PSAR" in psar_d.columns:
    ax_d.scatter(psar_d.index, psar_d["PSAR"].values, s=10, label="PSAR")

if np.isfinite(sup_d):
    ax_d.axhline(sup_d, linestyle="--", linewidth=0.9, alpha=0.8, label="Support")
if np.isfinite(res_d):
    ax_d.axhline(res_d, linestyle="--", linewidth=0.9, alpha=0.8, label="Resistance")

for name, val in (pivots_d or {}).items():
    if np.isfinite(val):
        ax_d.axhline(val, linestyle=":", linewidth=0.8, alpha=0.5)
        label_on_right(ax_d, val, name)

if show_fibs and fibs_d:
    for name, lv in fibs_d.items():
        ax_d.axhline(lv, linestyle=":", linewidth=0.7, alpha=0.35)
    overlay_fib_npx_signals(ax_d, hist_view, fib_buy_d, fib_sell_d)

bounce_d = find_band_bounce_signal(hist_view, up_d, lo_d, reg_slope_d)
if bounce_d is not None:
    annotate_crossover(ax_d, bounce_d["time"], bounce_d["price"], bounce_d["side"], note="Band bounce")

ax_d.set_title(
    f"{symbol} Daily | slope={fmt_slope(reg_slope_d)} | R²={fmt_r2(reg_r2_d)} | "
    f"revProb={fmt_pct(rev_prob_d, 1)}"
)
ax_d.legend(loc="best", ncol=2)
style_axes(ax_d)
st.pyplot(fig_d, use_container_width=True)

# =========================
# Daily NTD / NPX panel
# =========================
if show_ntd:
    st.subheader("Daily NTD / NPX")
    fig_ntd_d, ax_ntd_d = plt.subplots(figsize=(14, 4))
    ax_ntd_d.axhline(0, linewidth=0.8, alpha=0.5)

    if not ntd_daily.dropna().empty:
        ax_ntd_d.plot(ntd_daily.index, ntd_daily.values, label="NTD", linewidth=1.3)
        if shade_ntd:
            shade_ntd_regions(ax_ntd_d, ntd_daily)

    if show_npx_ntd and not npx_daily.dropna().empty:
        overlay_npx_on_ntd(ax_ntd_d, npx_daily, ntd_daily, mark_crosses=mark_npx_cross)

    draw_trend_direction_line(ax_ntd_d, ntd_daily, label="Trend Direction")

    if show_ntd_channel and np.isfinite(sup_d) and np.isfinite(res_d):
        sup_ser_d = pd.Series(sup_d, index=hist_view.index)
        res_ser_d = pd.Series(res_d, index=hist_view.index)
        overlay_inrange_on_ntd(ax_ntd_d, hist_view, sup_ser_d, res_ser_d)

    ax_ntd_d.set_ylim(-1.15, 1.15)
    ax_ntd_d.set_title("Daily Normalized Trend / Price")
    ax_ntd_d.legend(loc="best", ncol=3)
    style_axes(ax_ntd_d)
    st.pyplot(fig_ntd_d, use_container_width=True)

# =========================
# Hourly chart
# =========================
st.subheader("Hourly Price Chart")

if intraday is None or intraday.empty or "Close" not in intraday.columns:
    st.warning("No intraday data available.")
else:
    real_times = intraday.index
    x = np.arange(len(real_times), dtype=float)

    fig_h, ax_h = plt.subplots(figsize=(14, 6))
    close_h = _coerce_1d_series(intraday["Close"]).reindex(real_times)

    ax_h.plot(x, close_h.values, label="Close", linewidth=1.35)

    if not reg_h.dropna().empty:
        reg_h_plot = reg_h.reindex(real_times)
        ax_h.plot(x, reg_h_plot.values, label="Regression", linewidth=1.1, linestyle="--")

    if not up_h.dropna().empty and not lo_h.dropna().empty:
        up_h_plot = up_h.reindex(real_times)
        lo_h_plot = lo_h.reindex(real_times)
        ax_h.plot(x, up_h_plot.values, linewidth=0.9, linestyle=":")
        ax_h.plot(x, lo_h_plot.values, linewidth=0.9, linestyle=":")

    if show_hma and not hma_hour.dropna().empty:
        hma_h_plot = hma_hour.reindex(real_times)
        ax_h.plot(x, hma_h_plot.values, label=f"HMA({hma_period})", linewidth=1.0)

    if show_bbands and not bb_mid_h.dropna().empty:
        bb_mid_h_plot = bb_mid_h.reindex(real_times)
        bb_up_h_plot = bb_up_h.reindex(real_times)
        bb_lo_h_plot = bb_lo_h.reindex(real_times)
        ax_h.plot(x, bb_mid_h_plot.values, label="BB Mid", linewidth=0.95, alpha=0.9)
        ax_h.plot(x, bb_up_h_plot.values, linewidth=0.85, linestyle=":")
        ax_h.plot(x, bb_lo_h_plot.values, linewidth=0.85, linestyle=":")

    if show_ichi and not kijun_h.dropna().empty:
        kijun_h_plot = kijun_h.reindex(real_times)
        ax_h.plot(x, kijun_h_plot.values, label="Kijun", linewidth=0.95)

    if show_supertrend and not supertrend_h.empty and "ST" in supertrend_h.columns:
        st_h_plot = _coerce_1d_series(supertrend_h["ST"]).reindex(real_times)
        ax_h.plot(x, st_h_plot.values, label="Supertrend", linewidth=1.0)

    if show_psar and not psar_h.empty and "PSAR" in psar_h.columns:
        psar_h_plot = _coerce_1d_series(psar_h["PSAR"]).reindex(real_times)
        ax_h.scatter(x, psar_h_plot.values, s=10, label="PSAR")

    if np.isfinite(sup_h):
        ax_h.axhline(sup_h, linestyle="--", linewidth=0.9, alpha=0.8, label="Support")
    if np.isfinite(res_h):
        ax_h.axhline(res_h, linestyle="--", linewidth=0.9, alpha=0.8, label="Resistance")

    if show_fibs and fibs_h:
        for _, lv in fibs_h.items():
            ax_h.axhline(lv, linestyle=":", linewidth=0.7, alpha=0.35)
        overlay_fib_npx_signals(ax_h, pd.Series(close_h.values, index=x), fib_buy_h.set_axis(x[:len(fib_buy_h)]), fib_sell_h.set_axis(x[:len(fib_sell_h)]))
        bounce_h = find_band_bounce_signal(intraday_close, up_h, lo_h, reg_slope_h)
    if bounce_h is not None:
        bx = _map_times_to_bar_positions(real_times, [bounce_h["time"]])
        if bx:
            annotate_crossover(ax_h, bx[0], bounce_h["price"], bounce_h["side"], note="Band bounce")

    sessions = compute_session_lines(real_times)
    session_handles = []
    session_labels = []
    for label, tlist in sessions.items():
        pos = _map_times_to_bar_positions(real_times, tlist)
        if not pos:
            continue
        style_map = {
            "London Open": dict(ls="--", lw=1.0),
            "London Close": dict(ls=":", lw=1.0),
            "NY Open": dict(ls="--", lw=1.0),
            "NY Close": dict(ls=":", lw=1.0),
        }
        for p in pos:
            ax_h.axvline(p, alpha=0.28, **style_map.get(label, {}))
        session_handles.append(Line2D([0], [0], **style_map.get(label, {})))
        session_labels.append(label)

    news_df = fetch_yf_news(symbol, window_days=7)
    if not news_df.empty:
        news_pos = _map_times_to_bar_positions(real_times, news_df["time"].tolist())
        first = True
        for p in news_pos:
            ax_h.axvline(p, linestyle="-.", linewidth=0.9, alpha=0.18, label="News" if first else None)
            first = False

    _apply_compact_time_ticks(ax_h, real_times, n_ticks=8)
    ax_h.set_title(
        f"{symbol} Hourly | slope={fmt_slope(reg_slope_h)} | R²={fmt_r2(reg_r2_h)} | "
        f"revProb={fmt_pct(rev_prob_h, 1)} | current={fmt_price_val(current_price, digits=5 if mode == 'Forex' else 4)}"
    )
    style_axes(ax_h)

    handles, labels = ax_h.get_legend_handles_labels()
    if session_handles:
        handles.extend(session_handles)
        labels.extend(session_labels)
    if handles:
        ax_h.legend(handles, labels, loc="best", ncol=2)

    st.pyplot(fig_h, use_container_width=True)

# =========================
# Hourly NTD / NPX panel
# =========================
if show_nrsi:
    st.subheader("Hourly NTD / NPX")
    if intraday_close.empty:
        st.info("No hourly oscillator data available.")
    else:
        real_times = intraday_close.index
        x = np.arange(len(real_times), dtype=float)

        fig_ntd_h, ax_ntd_h = plt.subplots(figsize=(14, 4))
        ax_ntd_h.axhline(0, linewidth=0.8, alpha=0.5)

        ntd_h_plot = ntd_hour.reindex(real_times)
        npx_h_plot = npx_hour.reindex(real_times)

        if not ntd_h_plot.dropna().empty:
            ax_ntd_h.plot(x, ntd_h_plot.values, label="NTD", linewidth=1.25)
            if shade_ntd:
                yy = pd.Series(ntd_h_plot.values, index=x).dropna()
                if not yy.empty:
                    ax_ntd_h.fill_between(yy.index, 0, yy.values, where=(yy.values >= 0), alpha=0.15, interpolate=True)
                    ax_ntd_h.fill_between(yy.index, 0, yy.values, where=(yy.values < 0), alpha=0.10, interpolate=True)

        if show_npx_ntd and not npx_h_plot.dropna().empty:
            ax_ntd_h.plot(x, npx_h_plot.values, label="NPX", linewidth=1.0, alpha=0.95)
            if mark_npx_cross:
                up_mask, dn_mask = _strict_cross_series(npx_h_plot, ntd_h_plot)
                up_mask = up_mask.reindex(real_times, fill_value=False)
                dn_mask = dn_mask.reindex(real_times, fill_value=False)
                if up_mask.any():
                    up_x = np.where(up_mask.to_numpy())[0]
                    ax_ntd_h.scatter(up_x, npx_h_plot.iloc[up_x].values, marker="^", s=28, label="NPX↑NTD")
                if dn_mask.any():
                    dn_x = np.where(dn_mask.to_numpy())[0]
                    ax_ntd_h.scatter(dn_x, npx_h_plot.iloc[dn_x].values, marker="v", s=28, label="NPX↓NTD")

        if show_ntd_channel and np.isfinite(sup_h) and np.isfinite(res_h):
            sup_ser_h = pd.Series(sup_h, index=real_times)
            res_ser_h = pd.Series(res_h, index=real_times)
            state = channel_state_series(intraday_close, sup_ser_h, res_ser_h).reindex(real_times)
            in_range = (state == 0).fillna(False)
            above_r = (state > 0).fillna(False)
            below_s = (state < 0).fillna(False)

            for a, b in _true_spans(in_range):
                ax_ntd_h.axvspan(a, b, alpha=0.08, linewidth=0)

            enter_from_below = (below_s.shift(1, fill_value=False) & in_range)
            enter_from_above = (above_r.shift(1, fill_value=False) & in_range)

            if enter_from_below.any():
                idxs = np.where(enter_from_below.to_numpy())[0]
                ax_ntd_h.scatter(idxs, np.zeros(len(idxs)), marker="^", s=60, zorder=7, label="Enter from S")
            if enter_from_above.any():
                idxs = np.where(enter_from_above.to_numpy())[0]
                ax_ntd_h.scatter(idxs, np.zeros(len(idxs)), marker="v", s=60, zorder=7, label="Enter from R")

            last = state.dropna().iloc[-1] if state.dropna().shape[0] else np.nan
            if np.isfinite(last):
                state_txt = "Above R" if last > 0 else ("Below S" if last < 0 else "Inside Channel")
                ax_ntd_h.text(
                    0.995, 0.98, state_txt, transform=ax_ntd_h.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.75)
                )

        _apply_compact_time_ticks(ax_ntd_h, real_times, n_ticks=8)
        ax_ntd_h.set_ylim(-1.15, 1.15)
        ax_ntd_h.set_title("Hourly Normalized Trend / Price")
        ax_ntd_h.legend(loc="best", ncol=3)
        style_axes(ax_ntd_h)
        st.pyplot(fig_ntd_h, use_container_width=True)

# =========================
# MACD charts
# =========================
if show_macd:
    st.subheader("MACD")

    c1, c2 = st.columns(2)

    with c1:
        fig_md, ax_md = plt.subplots(figsize=(12, 4))
        if not macd_daily.dropna().empty:
            ax_md.plot(macd_daily.index, macd_daily.values, label="MACD", linewidth=1.1)
            ax_md.plot(macd_sig_daily.index, macd_sig_daily.values, label="Signal", linewidth=1.0)
            ax_md.bar(macd_hist_daily.index, macd_hist_daily.values, width=2.0, alpha=0.35, label="Hist")
        ax_md.axhline(0, linewidth=0.8, alpha=0.5)
        ax_md.set_title("Daily MACD")
        ax_md.legend(loc="best")
        style_axes(ax_md)
        st.pyplot(fig_md, use_container_width=True)

    with c2:
        fig_mh, ax_mh = plt.subplots(figsize=(12, 4))
        if not macd_hour.dropna().empty:
            real_times = macd_hour.index
            x = np.arange(len(real_times), dtype=float)
            ax_mh.plot(x, macd_hour.values, label="MACD", linewidth=1.1)
            ax_mh.plot(x, macd_sig_hour.reindex(real_times).values, label="Signal", linewidth=1.0)
            ax_mh.bar(x, macd_hist_hour.reindex(real_times).values, alpha=0.35, label="Hist")
            _apply_compact_time_ticks(ax_mh, real_times, n_ticks=8)
        ax_mh.axhline(0, linewidth=0.8, alpha=0.5)
        ax_mh.set_title("Hourly MACD")
        ax_mh.legend(loc="best")
        style_axes(ax_mh)
        st.pyplot(fig_mh, use_container_width=True)

# =========================
# News table
# =========================
st.subheader("Recent Yahoo News")
news_df = fetch_yf_news(symbol, window_days=7)
if news_df.empty:
    st.info("No recent Yahoo news found.")
else:
    show_news = news_df.copy()
    show_news["time"] = show_news["time"].dt.strftime("%Y-%m-%d %H:%M %Z")
    st.dataframe(show_news, use_container_width=True, hide_index=True)

# =========================
# Diagnostics / summary
# =========================
st.subheader("Diagnostics")

diag = {
    "Mode": mode,
    "Symbol": symbol,
    "Daily View": daily_view,
    "Hourly Window": hour_range,
    "Daily Bars": int(len(hist_view)),
    "Hourly Bars": int(len(intraday_close)),
    "Current Price": fmt_price_val(current_price, digits=5 if mode == "Forex" else 4),
    "Daily Last Close": fmt_price_val(daily_last_close, digits=5 if mode == "Forex" else 4),
    "Hourly Last Close": fmt_price_val(intraday_last_close, digits=5 if mode == "Forex" else 4),
    "Daily Slope": fmt_slope(reg_slope_d),
    "Daily R²": fmt_r2(reg_r2_d),
    "Daily Reversal Prob": fmt_pct(rev_prob_d, 1),
    "Hourly Slope": fmt_slope(reg_slope_h),
    "Hourly R²": fmt_r2(reg_r2_h),
    "Hourly Reversal Prob": fmt_pct(rev_prob_h, 1),
    "Support Daily": fmt_price_val(sup_d, digits=5 if mode == "Forex" else 4),
    "Resistance Daily": fmt_price_val(res_d, digits=5 if mode == "Forex" else 4),
    "Support Hourly": fmt_price_val(sup_h, digits=5 if mode == "Forex" else 4),
    "Resistance Hourly": fmt_price_val(res_h, digits=5 if mode == "Forex" else 4),
}

st.json(diag)
