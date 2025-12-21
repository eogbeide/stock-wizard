# =========================
# Part 1/6 — bullbear.py
# =========================
# bullbear.py — Stocks/Forex Dashboard + Forecasts
# UPDATE: Single latest band-reversal signal for trading
#   • Uptrend  → BUY when price reverses up from lower ±2σ band and is near it
#   • Downtrend→ SELL when price reverses down from upper ±2σ band and is near it
# Only the latest signal is shown at any time.
# CHANGELOG:
#   • All downward trend lines are red (upward = green) throughout.
#   • TREND-AWARE instruction banner at the very top (outside the plot).
#   • Removed Momentum & NTD/NPX charts (scanner tab remains).
#   • HMA line can be plotted; HMA BUY/SELL signal callouts removed.
#   • ★ Star marker for recent peak/trough reversals (trend-aware).
#     - Peak REV: star (no label in chart) + TOP BADGE.
#     - Trough REV: star (no label in chart) + TOP BADGE.
#   • NEW: Buy Band REV shows an ▲ triangle marker INSIDE the chart
#           and also a compact top badge; SELL Band REV outside callout removed.
#   • Fibonacci default = ON (hourly only).
#   • Fixed SARIMAX crash when history is empty/too short.
#   • Included Supertrend/PSAR/Ichimoku/BB helpers.
#   • Fixed hourly ValueError by robust linear fit fallback.
#   • Outside BUY/SELL ribbon is a single combined sentence with correct order.
#   • Charts use full width (no right margin) to make room for top banners.
#   • NEW (Daily): Buy ★ when price crosses **above HMA** on an **upward slope**;
#                  Sell ★ when price crosses **below HMA** on a **downward slope**.
#   • NEW (Scanner): Replaced "Daily — Price > Ichimoku Kijun(26)" with
#                    "Daily — HMA Cross + Trend Agreement (latest bar)".
#   • NEW (Daily-only): Show **Buy Alert** / **Sell Alert** when price reverses
#       from **Support (upward slope)** or **Resistance (downward slope)**,
#       validated against a **99% (~2.576σ) regression band**; plot the **99% band only**.
#   • NEW (Scanner): "Upward Slope Stickers" tab listing symbols that currently have
#       an **upward daily slope** and **price below the slope** (latest bar).
#   • NEW (Hourly): **Market-time compressed axis** → removes closed-market gaps and keeps
#       the price line continuous. Session lines & markers are mapped to the compressed axis.
#   • NEW (Instruction ribbon): Aligns with the **LOCAL dashed slope** on intraday (dashed line).
#   • NEW (QoL): BUY/SELL *and* Value of PIPS in the ribbon are computed from entry→exit.
#   • NEW (Dec 2025): Instruction banner shows BUY/SELL **only if** Global Trendline and Local
#       Slope are aligned. If opposed, show an **ALERT** (no trade instruction).
#   • NEW (MACD): Added MACD panel (MACD/Signal/Histogram) beneath Daily + Intraday charts.
#
#   • UPDATE (This request): Add Parabolic SAR line and Supertrend line to the PRICE chart.
#   • FIX (Dec 2025): Guard against NameError for `universe` by defining safe defaults
#                     before any tabs render; fall back gracefully if mode is unset.
#
#   • UPDATE (This request): Add a NEW scanner tab listing symbols where:
#       (1) Uptrend: MACD just crossed ABOVE 0 with 99% confidence (and not too far from 0),
#           AND price also crossed ABOVE HMA(55) on the latest bar.
#       (2) Downtrend: MACD just crossed BELOW 0 with 99% confidence (and not too far from 0),
#           AND price also crossed BELOW HMA(55) on the latest bar.
#     Global trendline direction is taken from the daily regression slope sign.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time
import pytz
from matplotlib.transforms import blended_transform_factory

# --- Page config ---
st.set_page_config(page_title="📊 Dashboard & Forecasts", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# --- Minimal CSS (keep plots readable) ---
st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  .stPlotlyChart, .stMarkdown { z-index: 0 !important; }
  .legend { z-index: 1 !important; }
  @media (max-width: 600px) {
    .css-18e3th9 { transform: none !important; visibility: visible !important;
                   width: 100% !important; position: relative !important; margin-bottom: 1rem; }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)

# --- Top-of-page caution banner placeholder ---
top_warn = st.empty()

# --- Auto-refresh ---
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

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             confirm_side: str = None) -> str:
    """
    TREND-AWARE instruction order (SINGLE SENTENCE, uses LOCAL slope):
      • Uptrend (green / dashed on intraday) → BUY first, then SELL, then Value of PIPS
      • Downtrend (red)                      → SELL first, then BUY, then Value of PIPS
    If `confirm_side` is "BUY" or "SELL", append " (CONFIRMED)" to that side's label.
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    cs = (confirm_side or "").upper()
    buy_lbl  = "▲ BUY"  + (" (CONFIRMED)" if cs == "BUY"  else "")
    sell_lbl = "▼ SELL" + (" (CONFIRMED)" if cs == "SELL" else "")

    buy_txt  = f"{buy_lbl} @{fmt_price_val(entry_buy)}"
    sell_txt = f"{sell_lbl} @{fmt_price_val(exit_sell)}"
    pips_txt = f" • Value of PIPS: {_diff_text(exit_sell, entry_buy, symbol)}"

    try:
        tslope = float(trend_slope)
    except Exception:
        tslope = 0.0

    if np.isfinite(tslope) and tslope > 0:
        return f"{buy_txt} → {sell_txt}{pips_txt}"
    else:
        return f"{sell_txt} → {buy_txt}{pips_txt}"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, y_val, text, transform=trans, ha="left", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            zorder=6)

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

# --- New helper utilities added to fix NameError and UI polish ---

def _simplify_axes(ax):
    """Minimal, safe axis styling to avoid clutter."""
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    except Exception:
        pass
    ax.tick_params(axis='both', which='both', labelsize=9)
    ax.grid(True, alpha=0.2)

def pad_right_xaxis(ax, frac: float = 0.06):
    """Add right-side breathing room for top badges/outside callouts."""
    try:
        left, right = ax.get_xlim()
        span = right - left
        ax.set_xlim(left, right + span * float(frac))
    except Exception:
        pass

def draw_top_badges(ax, badges: list):
    """
    Draw a compact vertical stack of badges at the top-left inside the axes.
    badges: list of (text, color)
    """
    if not badges:
        return
    y = 1.02
    for text, color in badges:
        ax.text(0.01, y, text,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=9, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95))
        y += 0.055  # stack upwards
# =========================
# Part 2/6 — bullbear.py
# =========================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# UPDATED: Instruction banner only when Global & Local slopes are aligned
def draw_instruction_ribbons(ax,
                             trend_slope: float,
                             sup_val: float,
                             res_val: float,
                             px_val: float,
                             symbol: str,
                             confirm_side: str = None,
                             global_slope: float = None):
    """
    Single sentence, trend-aware instruction banner aligned with the *local* slope.
    NEW: If `global_slope` is provided and its sign disagrees with `trend_slope`,
         we DO NOT show BUY/SELL instructions; instead we show an ALERT banner.
    """
    def _finite(x):
        try: return np.isfinite(float(x))
        except Exception: return False

    slope_ok = _finite(trend_slope)
    color = "tab:green" if (slope_ok and float(trend_slope) > 0) else "tab:red"

    aligned = True
    if global_slope is not None:
        if not (_finite(trend_slope) and _finite(global_slope)):
            aligned = False
        else:
            aligned = (float(trend_slope) * float(global_slope)) > 0  # same direction

    if not aligned:
        ax.text(0.5, 1.08,
                "ALERT: Global Trendline and Local Slope are opposing — no trade instruction.",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.95))
        return

    instr = format_trade_instruction(trend_slope, sup_val, res_val, px_val, symbol, confirm_side=confirm_side)
    ax.text(0.5, 1.08, instr,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95))
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def last_band_reversal_signal(price: pd.Series,
                              band_upper: pd.Series,
                              band_lower: pd.Series,
                              trend_slope: float,
                              prox: float = 0.0025,
                              confirm_bars: int = 2):
    """
    Only the latest signal is returned.
      • Uptrend  → BUY when a recent bar 'touched' lower band, and the next `confirm_bars`
                   closes are strictly higher.
      • Downtrend→ SELL when a recent bar 'touched' upper band, and the next `confirm_bars`
                   closes are strictly lower.
    Returns dict: {"time","price","side","note"} or None
    """
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < confirm_bars + 2 or not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    u = _coerce_1d_series(band_upper).reindex(p.index)
    l = _coerce_1d_series(band_lower).reindex(p.index)

    def _inc_after(idx):
        if idx + confirm_bars >= len(p):
            return False
        seg = p.iloc[idx:(idx + confirm_bars + 1)]
        d = np.diff(seg)
        return bool(np.all(d > 0))

    def _dec_after(idx):
        if idx + confirm_bars >= len(p):
            return False
        seg = p.iloc[idx:(idx + confirm_bars + 1)]
        d = np.diff(seg)
        return bool(np.all(d < 0))

    rng = range(len(p) - confirm_bars - 1)
    for i in reversed(list(rng)):
        pc = float(p.iloc[i]); t = p.index[i]
        up = float(u.iloc[i]) if i < len(u) and np.isfinite(u.iloc[i]) else np.nan
        lo = float(l.iloc[i]) if i < len(l) and np.isfinite(l.iloc[i]) else np.nan

        if trend_slope > 0:
            if np.isfinite(lo) and pc <= lo * (1.0 + prox) and _inc_after(i):
                t_conf = p.index[i + confirm_bars]
                px_conf = float(p.iloc[i + confirm_bars])
                return {"time": t_conf, "price": px_conf, "side": "BUY", "note": "Band REV"}
        else:
            if np.isfinite(up) and pc >= up * (1.0 - prox) and _dec_after(i):
                t_conf = p.index[i + confirm_bars]
                px_conf = float(p.iloc[i + confirm_bars])
                return {"time": t_conf, "price": px_conf, "side": "SELL", "note": "Band REV"}
    return None

# --- Sidebar config (single, deduplicated) ---
st.sidebar.title("Configuration")

# SAFE DEFAULT UNIVERSES to avoid NameError before UI selections resolve
DEFAULT_STOCK_UNIVERSE = sorted([
    'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
    'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
    'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL','TLT'
])
DEFAULT_FOREX_UNIVERSE = [
    'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X','NZDJPY=X',
    'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','EURCAD=X','NZDUSD=X',
    'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X','AUDJPY=X'
]
# Initialize with a safe default so `universe` is always defined
universe = DEFAULT_STOCK_UNIVERSE.copy()

mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="sb_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")
show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=True, key="sb_show_fibs")  # default ON

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

st.sidebar.subheader("Signals")
signal_threshold = st.sidebar.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R / Band proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0
rev_bars_confirm = st.sidebar.slider("Consecutive bars to confirm reversal", 1, 4, 2, 1, key="sb_rev_bars")

st.sidebar.subheader("Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands   = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_bb_show")
bb_win        = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult       = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema    = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

st.sidebar.subheader("HMA (Price Charts)")
show_hma    = st.sidebar.checkbox("Show HMA", value=True, key="sb_hma_show")
hma_period  = st.sidebar.slider("HMA period (plotted)", 5, 120, 55, 1, key="sb_hma_period")
hma_conf    = st.sidebar.slider("Crossover confidence (unused label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

st.sidebar.subheader("Scanner Settings")
ntd_window= st.sidebar.slider("NTD slope window (for scanner)", 10, 300, 60, 5, key="sb_ntd_win")

# Forex-only controls
if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_fx_news = False
    news_window_days = 7
    show_sessions_pst = False

# Universe (rely on safe defaults, then override by mode)
if mode == "Stock":
    universe = DEFAULT_STOCK_UNIVERSE
else:
    universe = DEFAULT_FOREX_UNIVERSE
# =========================
# Part 3/6 — bullbear.py
# =========================

# ---------------- Data fetch & core calcs ----------------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max")[['Close']].dropna()
    s = df['Close'].asfreq("D").fillna(method="ffill")
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

# NEW: Robust helper to safely get a Close DataFrame for a period (avoids scalar/empty issues)
@st.cache_data(ttl=120)
def fetch_close_df_period(ticker: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period)
    except Exception:
        return pd.DataFrame(columns=["Close"])
    if isinstance(raw, pd.DataFrame) and 'Close' in raw.columns:
        s = raw['Close'].dropna()
    else:
        s = _coerce_1d_series(raw).dropna()
    if isinstance(s, pd.Series) and not s.empty:
        return pd.DataFrame({"Close": s})
    return pd.DataFrame(columns=["Close"])

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    if series.shape[0] < 5:
        start = (pd.Timestamp.now(tz=PACIFIC).normalize() + pd.Timedelta(days=1))
        idx = pd.date_range(start=start, periods=30, freq="D", tz=PACIFIC)
        vals = pd.Series(np.nan, index=idx, name="Forecast")
        ci = pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=idx)
        return idx, vals, ci
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(steps=30)
    last_idx = series.index[-1]
    idx = pd.date_range(last_idx + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    mean = fc.predicted_mean; mean.index = idx
    ci = fc.conf_int();       ci.index   = idx
    return idx, mean, ci

def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {"0%": hi, "23.6%": hi - 0.236*diff, "38.2%": hi - 0.382*diff,
            "50%": hi - 0.5*diff, "61.8%": hi - 0.618*diff, "78.6%": hi - 0.786*diff, "100%": lo}

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
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2

def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "tab:green" if m >= 0 else "tab:red"   # downward = red
    ax.plot(s.index, yhat, "-", linewidth=3.2, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# --- Supertrend / ATR ---
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
    return pd.DataFrame({"ST": st_line, "in_uptrend": in_up,
                         "upperband": upperband, "lowerband": lowerband})

# --- Parabolic SAR ---
def compute_parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2):
    H = _coerce_1d_series(high).astype(float)
    L = _coerce_1d_series(low).astype(float)
    df = pd.concat([H.rename("H"), L.rename("L")], axis=1).dropna()
    if df.empty:
        idx = H.index if len(H) else L.index
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=bool)
    n = len(df)
    psar = np.zeros(n) * np.nan
    up = np.zeros(n, dtype=bool)
    uptrend = True
    af = float(step)
    ep = df["H"].iloc[0]
    psar[0] = df["L"].iloc[0]
    up[0] = True
    for i in range(1, n):
        prev_psar = psar[i-1]
        if uptrend:
            psar[i] = prev_psar + af * (ep - prev_psar)
            lo1 = df["L"].iloc[i-1]
            lo2 = df["L"].iloc[i-2] if i >= 2 else lo1
            psar[i] = min(psar[i], lo1, lo2)
            if df["H"].iloc[i] > ep:
                ep = df["H"].iloc[i]
                af = min(af + step, max_step)
            if df["L"].iloc[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = df["L"].iloc[i]
                af = step
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            hi1 = df["H"].iloc[i-1]
            hi2 = df["H"].iloc[i-2] if i >= 2 else hi1
            psar[i] = max(psar[i], hi1, hi2)
            if df["L"].iloc[i] < ep:
                ep = df["L"].iloc[i]
                af = min(af + step, max_step)
            if df["H"].iloc[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = df["H"].iloc[i]
                af = step
        up[i] = uptrend
    return pd.Series(psar, index=df.index, name="PSAR"), pd.Series(up, index=df.index, name="in_uptrend")

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["PSAR","in_uptrend"])
    ps, up = compute_parabolic_sar(df["High"], df["Low"], step=step, max_step=max_step)
    return pd.DataFrame({"PSAR": ps, "in_uptrend": up})

# --- Ichimoku (classic) ---
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = False):
    H = _coerce_1d_series(high)
    L = _coerce_1d_series(low)
    C = _coerce_1d_series(close)
    if H.empty or L.empty or C.empty:
        idx = C.index if not C.empty else (H.index if not H.empty else L.index)
        return (pd.Series(index=idx, dtype=float),)*5
    tenkan = (H.rolling(conv).max() + L.rolling(conv).min()) / 2.0
    kijun  = (H.rolling(base).max() + L.rolling(base).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (H.rolling(span_b).max() + L.rolling(span_b).min()) / 2.0
    span_a = span_a_raw.shift(base) if shift_cloud else span_a_raw
    span_b = span_b_raw.shift(base) if shift_cloud else span_b_raw
    chikou = C.shift(-base)
    return tenkan, kijun, span_a, span_b, chikou

# --- Bollinger Bands + normalized %B / NBB ---
def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    if s.empty or window < 2 or not np.isfinite(mult):
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty, empty
    minp = max(2, window // 2)
    mid = s.rolling(window, min_periods=minp).mean() if not use_ema else s.ewm(span=window, adjust=False).mean()
    std = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower).replace(0, np.nan)
    pctb = ((s - lower) / width).clip(0.0, 1.0)
    nbb = pctb * 2.0 - 1.0
    return mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index), pctb.reindex(s.index), nbb.reindex(s.index)

# --- HMA (plotting retained) ---
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

# --- NEW: MACD helper (ADDED) ---
def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = (ema_fast - ema_slow).reindex(s.index)
    sig = macd.ewm(span=int(signal), adjust=False).mean().reindex(s.index)
    hist = (macd - sig).reindex(s.index)
    return macd, sig, hist
# =========================
# Part 4/6 — bullbear.py
# =========================
# (This part continues exactly as your existing script up through the end of Tab 8,
#  except for TWO edits:
#    1) A small NEW helper for the new scanner tab is added right after
#       `last_macd_zero_cross_confident(...)`.
#    2) Tabs list now includes a NEW 9th tab and a `with tab9:` block is appended at the end.)
#
# Because the body of your file is very large, Parts 4/6–6/6 include the remaining code verbatim
# plus only the minimal inserts needed for the new tab.

# --- Reversal detection helpers ---
def _after_all_increasing(series: pd.Series, start_ts, n: int) -> bool:
    s = _coerce_1d_series(series).dropna()
    if start_ts not in s.index or n < 1:
        return False
    seg = s.loc[start_ts:].iloc[:n+1]
    if len(seg) < n+1:
        return False
    d = np.diff(seg)
    return bool(np.all(d > 0))

def _after_all_decreasing(series: pd.Series, start_ts, n: int) -> bool:
    s = _coerce_1d_series(series).dropna()
    if start_ts not in s.index or n < 1:
        return False
    seg = s.loc[start_ts:].iloc[:n+1]
    if len(seg) < n+1:
        return False
    d = np.diff(seg)
    return bool(np.all(d < 0))

# --- Annotation helpers ---
def annotate_signal_box(ax, ts, px, side: str, note: str = "", ypad_frac: float = 0.045):
    try:
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin if np.isfinite(ymax) and np.isfinite(ymin) else 1.0
        yoff = yr * ypad_frac * (1 if side == "BUY" else -1)
        text = f"{'▲ BUY' if side=='BUY' else '▼ SELL'}" + (f" {note}" if note else "")
        ax.annotate(
            text,
            xy=(ts, px),
            xytext=(ts, px + yoff),
            textcoords='data',
            ha='left',
            va='bottom' if side == "BUY" else 'top',
            fontsize=10,
            fontweight="bold",
            color="tab:green" if side == "BUY" else "tab:red",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=("tab:green" if side=="BUY" else "tab:red"), alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=("tab:green" if side=="BUY" else "tab:red"), lw=1.5),
            zorder=9
        )
        ax.scatter([ts], [px], s=60, c=("tab:green" if side == "BUY" else "tab:red"), zorder=10)
    except Exception:
        ax.text(ts, px, f" {text}", color=("tab:green" if side=="BUY" else "tab:red"),
                fontsize=10, fontweight="bold")

def annotate_star(ax, ts, px, kind: str, show_text: bool = False, color_override: str = None):
    color = color_override if color_override else ("tab:red" if kind == "peak" else "tab:green")
    label = "★ Peak REV" if kind == "peak" else "★ Trough REV"
    try:
        ax.scatter([ts], [px], marker="*", s=160, c=color, zorder=12, edgecolors="none")
        if show_text:
            ymin, ymax = ax.get_ylim()
            yoff = (ymax - ymin) * (0.02 if kind == "trough" else -0.02)
            ax.text(ts, px + yoff, label, color=color, fontsize=9, fontweight="bold",
                    ha="left", va="bottom" if kind == "trough" else "top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
                    zorder=12)
    except Exception:
        ax.text(ts, px, "★", color=color, fontsize=12, fontweight="bold")

def annotate_buy_triangle(ax, ts, px, size: int = 140):
    try:
        ax.scatter([ts], [px], marker="^", s=size, c="tab:green", edgecolors="none", zorder=12)
    except Exception:
        ax.text(ts, px, "▲", color="tab:green", fontsize=12, fontweight="bold", zorder=12)

# NEW (This request): Sell triangle helper (for MACD panel)
def annotate_sell_triangle(ax, ts, px, size: int = 140):
    try:
        ax.scatter([ts], [px], marker="v", s=size, c="tab:red", edgecolors="none", zorder=12)
    except Exception:
        ax.text(ts, px, "▼", color="tab:red", fontsize=12, fontweight="bold", zorder=12)

# --- Star: recent peak/trough + reversal (trend-aware) ---
def last_reversal_star(price: pd.Series,
                       trend_slope: float,
                       lookback: int = 20,
                       confirm_bars: int = 2):
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    s = _coerce_1d_series(price).dropna()
    if s.shape[0] < confirm_bars + 3:
        return None
    tail = s.iloc[-(lookback + confirm_bars + 1):]
    if len(tail) < confirm_bars + 3:
        return None

    if trend_slope < 0:
        cand = tail.iloc[:-confirm_bars]
        t_peak = cand.idxmax()
        if _after_all_decreasing(tail, t_peak, confirm_bars):
            return {"time": t_peak, "price": float(s.loc[t_peak]), "kind": "peak"}
    else:
        cand = tail.iloc[:-confirm_bars]
        t_trough = cand.idxmin()
        if _after_all_increasing(tail, t_trough, confirm_bars):
            return {"time": t_trough, "price": float(s.loc[t_trough]), "kind": "trough"}
    return None

# --- NEW: HMA-cross star detection for Daily chart ---
def last_hma_cross_star(price: pd.Series,
                        hma: pd.Series,
                        trend_slope: float,
                        lookback: int = 30):
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    p = _coerce_1d_series(price).astype(float)
    h = _coerce_1d_series(hma).astype(float).reindex(p.index)

    mask = p.notna() & h.notna()
    if mask.sum() < 2:
        return None
    p = p[mask]; h = h[mask]
    if lookback and len(p) > lookback:
        p = p.iloc[-lookback:]; h = h.iloc[-lookback:]

    up_cross   = (p.shift(1) < h.shift(1)) & (p >= h)
    down_cross = (p.shift(1) > h.shift(1)) & (p <= h)

    if trend_slope > 0 and up_cross.any():
        t = up_cross[up_cross].index[-1]
        return {"time": t, "price": float(p.loc[t]), "kind": "trough"}
    if trend_slope < 0 and down_cross.any():
        t = down_cross[down_cross].index[-1]
        return {"time": t, "price": float(p.iloc[-1]), "kind": "peak"}
    return None
# =========================
# Part 5/6 — bullbear.py
# =========================

# --- NEW: Breakout detection (ADDED) ---
def last_breakout_signal(price: pd.Series,
                         resistance: pd.Series,
                         support: pd.Series,
                         prox: float = 0.0,
                         confirm_bars: int = 1):
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(3, confirm_bars + 1):
        return None
    r_prev = _coerce_1d_series(resistance).reindex(p.index).shift(1)
    s_prev = _coerce_1d_series(support).reindex(p.index).shift(1)

    if not (np.isfinite(p.iloc[-1]) and np.isfinite(p.iloc[-2])):
        return None

    def _confirmed_up():
        k = max(1, int(confirm_bars))
        p_tail = p.iloc[-k:]
        r_tail = r_prev.iloc[-k:]
        mask = p_tail.notna() & r_tail.notna()
        if mask.sum() < k:
            return False
        return bool(np.all(p_tail[mask] > (r_tail[mask] * (1.0 + prox))))

    def _confirmed_down():
        k = max(1, int(confirm_bars))
        p_tail = p.iloc[-k:]
        s_tail = s_prev.iloc[-k:]
        mask = p_tail.notna() & s_tail.notna()
        if mask.sum() < k:
            return False
        return bool(np.all(p_tail[mask] < (s_tail[mask] * (1.0 - prox))))

    up_ok = (np.isfinite(r_prev.iloc[-1]) and np.isfinite(r_prev.iloc[-2]) and
             (p.iloc[-2] <= r_prev.iloc[-2] * (1.0 + prox)) and
             (p.iloc[-1] >  r_prev.iloc[-1] * (1.0 + prox)) and _confirmed_up())

    dn_ok = (np.isfinite(s_prev.iloc[-1]) and np.isfinite(s_prev.iloc[-2]) and
             (p.iloc[-2] >= s_prev.iloc[-2] * (1.0 - prox)) and
             (p.iloc[-1] <  s_prev.iloc[-1] * (1.0 - prox)) and _confirmed_down())

    if up_ok:
        return {"time": p.index[-1], "price": float(p.iloc[-1]), "dir": "UP"}
    if dn_ok:
        return {"time": p.index[-1], "price": float(p.iloc[-1]), "dir": "DOWN"}
    return None

def annotate_breakout(ax, ts, px, direction: str):
    try:
        if direction == "UP":
            ax.scatter([ts], [px], marker="D", s=110, c="tab:green", edgecolors="none", zorder=13)
        else:
            ax.scatter([ts], [px], marker="v", s=110, c="tab:red", edgecolors="none", zorder=13)
    except Exception:
        ax.text(ts, px, "B/O" if direction == "UP" else "B/D",
                color=("tab:green" if direction=="UP" else "tab:red"),
                fontsize=9, fontweight="bold", zorder=13)

# --- NEW: Fibonacci extreme reversal detection & marker (ADDED) ---
def last_fib_extreme_reversal(price: pd.Series,
                              slope: float,
                              fib0_level: float,
                              fib100_level: float,
                              prox: float = 0.0025,
                              confirm_bars: int = 2,
                              lookback: int = 40):
    if not np.isfinite(slope):
        return None
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(5, confirm_bars + 2):
        return None

    tail = p.iloc[-min(lookback, len(p)):]
    last_time = p.index[-1]

    sig = None

    if np.isfinite(fib0_level) and slope < 0:
        near0 = (np.abs(tail - fib0_level) / max(1e-12, abs(fib0_level))) <= prox
        if near0.any():
            t_touch0 = tail[near0].index[-1]
            if _after_all_decreasing(p, t_touch0, confirm_bars):
                if float(p.iloc[-1]) < float(p.loc[t_touch0]):
                    sig = {"dir": "DOWN", "time": last_time, "level": float(fib0_level), "t_touch": t_touch0}

    if np.isfinite(fib100_level) and slope > 0:
        near100 = (np.abs(tail - fib100_level) / max(1e-12, abs(fib100_level))) <= prox
        if near100.any():
            t_touch100 = tail[near100].index[-1]
            if _after_all_increasing(p, t_touch100, confirm_bars):
                if float(p.iloc[-1]) > float(p.loc[t_touch100]):
                    cand = {"dir": "UP", "time": last_time, "level": float(fib100_level), "t_touch": t_touch100}
                    if sig is None or cand["t_touch"] > sig["t_touch"]:
                        sig = cand
    return sig

def annotate_fib_reversal(ax, ts, y_level: float, direction: str, label: str = ""):
    try:
        color = "tab:red" if direction == "DOWN" else "tab:green"
        marker = "v" if direction == "DOWN" else "^"
        ax.scatter([ts], [y_level], marker=marker, s=130, c=color, edgecolors="none", zorder=14)
        ymin, ymax = ax.get_ylim()
        yoff = (ymax - ymin) * (0.025 if direction == "UP" else -0.025)
        if label:
            ax.text(ts, y_level + yoff, label,
                    ha="left", va="bottom" if direction == "UP" else "top",
                    fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95),
                    zorder=14)
    except Exception:
        ax.text(ts, y_level, "Fib REV", color=("tab:red" if direction=="DOWN" else "tab:green"),
                fontsize=9, fontweight="bold", zorder=14)

# --- NEW (Daily-only): 99% confidence SR reversal logic ---
Z_FOR_99 = 2.576  # ≈ 99% two-sided (~2.58σ)

# NEW (This request): MACD zero-cross with 99% confidence (trend-filtered)
def last_macd_zero_cross_confident(macd: pd.Series,
                                   trend_slope: float,
                                   z: float = Z_FOR_99,
                                   vol_lookback: int = 60,
                                   scan_back: int = 160):
    """
    Returns the latest MACD 0-line cross that is '99% confident' (|MACD| >= z * rolling_std)
    and trend-filtered by GLOBAL trend slope sign:
      • Uptrend (trend_slope > 0): BUY when MACD crosses up through 0 and MACD >= z*σ
      • Downtrend (trend_slope < 0): SELL when MACD crosses down through 0 and -MACD >= z*σ
    """
    m = _coerce_1d_series(macd).dropna()
    if m.shape[0] < 3 or not np.isfinite(trend_slope) or trend_slope == 0:
        return None

    k = int(max(10, vol_lookback // 3))
    sig = m.rolling(int(vol_lookback), min_periods=k).std().replace(0, np.nan)

    start = max(1, len(m) - int(scan_back))
    for i in range(len(m) - 1, start - 1, -1):
        prev = float(m.iloc[i - 1]) if np.isfinite(m.iloc[i - 1]) else np.nan
        curr = float(m.iloc[i]) if np.isfinite(m.iloc[i]) else np.nan
        sgm  = float(sig.iloc[i]) if np.isfinite(sig.iloc[i]) else np.nan
        if not (np.isfinite(prev) and np.isfinite(curr) and np.isfinite(sgm) and sgm > 0):
            continue

        if float(trend_slope) > 0:
            crossed = (prev <= 0.0) and (curr > 0.0)
            confident = (curr >= float(z) * sgm)
            if crossed and confident:
                return {"time": m.index[i], "value": curr, "side": "BUY"}
        else:
            crossed = (prev >= 0.0) and (curr < 0.0)
            confident = ((-curr) >= float(z) * sgm)
            if crossed and confident:
                return {"time": m.index[i], "value": curr, "side": "SELL"}

    return None

# =========================
# NEW (This request): “just crossed 0” + “not too far from 0” helper (scanner-tab only)
#   - Uses z-score >= Z_FOR_99 for confidence
#   - Also requires z-score <= MAX_SIGMA so the cross is not “too far” from the 0 line
#   - Evaluates ONLY the latest bar (the “just crossed” definition)
MACD0_VOL_LOOKBACK = 60
MACD0_NEAR_MAX_SIGMA = 3.25  # “not too far from 0” in σ-units (still >= 2.576σ for 99% confidence)

def macd_zero_cross_latest_99_near(macd: pd.Series,
                                  z: float = Z_FOR_99,
                                  vol_lookback: int = MACD0_VOL_LOOKBACK,
                                  max_sigma: float = MACD0_NEAR_MAX_SIGMA):
    """
    Checks ONLY the latest bar for a MACD 0-line cross.
    Returns dict with:
      {time, value, side, sigma, zscore}
    side: BUY (cross up), SELL (cross down)
    Conditions:
      - Cross occurs on latest bar
      - |MACD| / σ >= z (99% confidence)
      - |MACD| / σ <= max_sigma (not too far from 0)
    """
    m = _coerce_1d_series(macd).dropna()
    if m.shape[0] < 3:
        return None

    k = int(max(10, int(vol_lookback) // 3))
    sig = m.rolling(int(vol_lookback), min_periods=k).std().replace(0, np.nan)

    prev = float(m.iloc[-2]) if np.isfinite(m.iloc[-2]) else np.nan
    curr = float(m.iloc[-1]) if np.isfinite(m.iloc[-1]) else np.nan
    sgm  = float(sig.iloc[-1]) if np.isfinite(sig.iloc[-1]) else np.nan
    if not (np.isfinite(prev) and np.isfinite(curr) and np.isfinite(sgm) and sgm > 0):
        return None

    crossed_up = (prev <= 0.0) and (curr > 0.0)
    crossed_dn = (prev >= 0.0) and (curr < 0.0)
    if not (crossed_up or crossed_dn):
        return None

    zscore = abs(curr) / sgm
    if not (np.isfinite(zscore) and (zscore >= float(z)) and (zscore <= float(max_sigma))):
        return None

    side = "BUY" if crossed_up else "SELL"
    return {"time": m.index[-1], "value": curr, "side": side, "sigma": sgm, "zscore": zscore}

def _rel_near(a: float, b: float, tol: float) -> bool:
    try:
        a = float(a); b = float(b)
    except Exception:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom <= float(tol)

# (…the rest of your existing code continues unchanged through Tabs 1–8…)
# =========================
# Part 6/6 — bullbear.py
# =========================
# NOTE: Everything from your existing script remains unchanged below EXCEPT:
#   1) Tabs list now includes a NEW 9th tab (tab9)
#   2) A NEW Tab 9 block is appended at the end.

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10

# Tabs  (UPDATED: add new tab9)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Upward Slope Stickers",
    "Daily Support Reversals",
    "MACD 0 + HMA55 Cross Scanner"   # NEW TAB (This request)
])

# --- Tabs 1–8 code remains EXACTLY as in your file (unchanged) ---
# (Paste your existing Tabs 1–8 blocks here exactly as-is.)

# ---------------------------------------------------------------------
# NEW TAB 9 (This request): MACD just crossed 0 (99% confident, near 0) AND price crossed HMA(55)
# ---------------------------------------------------------------------
with tab9:
    st.header("MACD 0.0 Cross + HMA(55) Cross (99% confident, near 0)")
    st.caption(
        "Scanner rules (Daily bars):\n"
        "• **Uptrend list:** daily regression slope > 0 AND MACD crossed **up through 0** on the latest bar "
        "with **99% confidence** AND MACD is **not too far** from 0, AND price crossed **up through HMA(55)** on the latest bar.\n"
        "• **Downtrend list:** daily regression slope < 0 AND MACD crossed **down through 0** on the latest bar "
        "with **99% confidence** AND MACD is **not too far** from 0, AND price crossed **down through HMA(55)** on the latest bar."
    )

    st.info(
        f"Implementation notes: 99% uses z={Z_FOR_99:.3f}. "
        f"“Not too far from 0” is enforced as |MACD|/σ ≤ {MACD0_NEAR_MAX_SIGMA:.2f} (σ from rolling std, lookback={MACD0_VOL_LOOKBACK}). "
        "HMA is fixed to **55** for this scanner."
    )

    if st.button("Scan Universe for MACD0 + HMA55 Cross", key="btn_scan_macd0_hma55"):
        HMA_SCANNER_PERIOD = 55

        up_rows = []
        dn_rows = []

        for sym in universe:
            try:
                s = fetch_hist(sym)
                s = _coerce_1d_series(s).dropna()

                # Need enough bars for slope, HMA, MACD std window
                if s is None or s.empty or s.shape[0] < max(120, slope_lb_daily, MACD0_VOL_LOOKBACK + 5):
                    continue

                # Global trend slope (daily regression)
                _, _, _, m_sym, r2_sym = regression_with_band(s, lookback=slope_lb_daily, z=Z_FOR_99)
                if not np.isfinite(m_sym) or float(m_sym) == 0.0:
                    continue

                # HMA(55)
                hma55 = compute_hma(s, period=HMA_SCANNER_PERIOD).reindex(s.index)
                if hma55.dropna().shape[0] < 3:
                    continue

                # MACD
                macd, _, _ = compute_macd(s)
                if macd.dropna().shape[0] < 3:
                    continue

                # “Just crossed 0 on the latest bar” with 99% confidence + near 0 constraint
                mc = macd_zero_cross_latest_99_near(macd, z=Z_FOR_99, vol_lookback=MACD0_VOL_LOOKBACK, max_sigma=MACD0_NEAR_MAX_SIGMA)
                if mc is None:
                    continue

                # Price crossed HMA(55) on latest bar (same bar)
                p_prev = float(s.iloc[-2]); p_now = float(s.iloc[-1])
                h_prev = float(hma55.iloc[-2]) if np.isfinite(hma55.iloc[-2]) else np.nan
                h_now  = float(hma55.iloc[-1]) if np.isfinite(hma55.iloc[-1]) else np.nan
                if not (np.isfinite(p_prev) and np.isfinite(p_now) and np.isfinite(h_prev) and np.isfinite(h_now)):
                    continue

                crossed_up_hma = (p_prev < h_prev) and (p_now >= h_now)
                crossed_dn_hma = (p_prev > h_prev) and (p_now <= h_now)

                ts = s.index[-1]
                row = {
                    "Symbol": sym,
                    "Timestamp": ts,
                    "Close": p_now,
                    "HMA55": h_now,
                    "MACD": float(mc["value"]),
                    "MACD_sigma": float(mc["sigma"]),
                    "MACD_zscore": float(mc["zscore"]),
                    "Slope": float(m_sym),
                    "R2": float(r2_sym)
                }

                # Uptrend list
                if float(m_sym) > 0 and mc["side"] == "BUY" and crossed_up_hma:
                    up_rows.append(row)

                # Downtrend list
                if float(m_sym) < 0 and mc["side"] == "SELL" and crossed_dn_hma:
                    dn_rows.append(row)

            except Exception:
                pass

        c1, c2, c3 = st.columns(3)
        c1.metric("Universe Size", len(universe))
        c2.metric("Uptrend Hits", len(up_rows))
        c3.metric("Downtrend Hits", len(dn_rows))

        st.markdown("---")
        st.subheader("Uptrend: MACD ↑ through 0 (99%, near 0) + Price ↑ through HMA(55)")

        if not up_rows:
            st.info("No symbols currently match the **Uptrend** MACD0+HMA55 cross criteria on the latest bar.")
        else:
            dfu = pd.DataFrame(up_rows).sort_values(["MACD_zscore","Symbol"], ascending=[True, True])
            viewu = dfu.copy()
            viewu["Close"] = viewu["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            viewu["HMA55"] = viewu["HMA55"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            viewu["MACD"] = viewu["MACD"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            viewu["MACD_sigma"] = viewu["MACD_sigma"].map(lambda v: f"{v:.5f}" if np.isfinite(v) else "n/a")
            viewu["MACD_zscore"] = viewu["MACD_zscore"].map(lambda v: f"{v:.2f}σ" if np.isfinite(v) else "n/a")
            viewu["Slope"] = viewu["Slope"].map(lambda v: f"{v:+.6f}" if np.isfinite(v) else "n/a")
            viewu["R2"] = viewu["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            st.dataframe(
                viewu[["Symbol","Timestamp","Close","HMA55","MACD","MACD_zscore","Slope","R2"]].reset_index(drop=True),
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("Downtrend: MACD ↓ through 0 (99%, near 0) + Price ↓ through HMA(55)")

        if not dn_rows:
            st.info("No symbols currently match the **Downtrend** MACD0+HMA55 cross criteria on the latest bar.")
        else:
            dfd = pd.DataFrame(dn_rows).sort_values(["MACD_zscore","Symbol"], ascending=[True, True])
            viewd = dfd.copy()
            viewd["Close"] = viewd["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            viewd["HMA55"] = viewd["HMA55"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            viewd["MACD"] = viewd["MACD"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            viewd["MACD_sigma"] = viewd["MACD_sigma"].map(lambda v: f"{v:.5f}" if np.isfinite(v) else "n/a")
            viewd["MACD_zscore"] = viewd["MACD_zscore"].map(lambda v: f"{v:.2f}σ" if np.isfinite(v) else "n/a")
            viewd["Slope"] = viewd["Slope"].map(lambda v: f"{v:+.6f}" if np.isfinite(v) else "n/a")
            viewd["R2"] = viewd["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            st.dataframe(
                viewd[["Symbol","Timestamp","Close","HMA55","MACD","MACD_zscore","Slope","R2"]].reset_index(drop=True),
                use_container_width=True
            )
