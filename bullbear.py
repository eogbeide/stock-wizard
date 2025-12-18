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
#           and also a compact top badge; SELL Band REV remains an outside callout.
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
#   • NEW (Daily): HMA Cross colors → Buy = Black, Sell = Blue (badges and stars).
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
#       Slope are aligned. If opposed, show an **ALERT** (no trade instruction). Daily chart also
#       includes the **Price Reversed** outside marker.
#   • NEW (Dec 2025): Hourly chart: removed ±2σ lines; added Buy/Sell signal when
#       HMA(55) cross occurs AFTER support/resistance reversal in agreement with GLOBAL slope.
#   • NEW (Dec 2025): **De-clutter HMA Buy signal** — HMA Buy/Sell callouts are now
#       **outside the chart**, auto-stacked at **top-right**, with a connector line from
#       the entry point; an optional faint vertical entry line is also drawn.
#   • NEW (Dec 2025): Daily chart also shows the same **outside HMA(55) Buy/Sell** callouts.
#   • NEW (Dec 2025): Added **HMA 55 Reversals** tab: lists symbols where price **reversed**
#       (peak/trough) and later **crossed HMA(55)** in agreement with the reversal side.

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
         we DO NOT show BUY/SELL instructions; instead we show an **ALERT** banner.
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
        # ALERT banner only (no BUY/SELL instruction sentence)
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

# Universe
if mode == "Stock":
    universe = sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL','TLT'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X','NZDJPY=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','EURCAD=X','NZDUSD=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X','AUDJPY=X'
    ]

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

# ===================== OUTSIDE CALLOUTS (NO CLUTTER) ======================
def _next_outside_box_position(ax):
    """
    Returns a (x_text, y_text) position in *figure* coordinates to place
    an outside callout, auto-stacked at the top-right to avoid overlaps.
    """
    if not hasattr(ax, "_outside_box_n"):
        ax._outside_box_n = 0
    n = ax._outside_box_n
    ax._outside_box_n += 1
    fig = ax.figure
    ax_bbox = ax.get_position()  # in figure coords
    x_text = float(min(0.98, ax_bbox.x1 - 0.02))          # right side column
    y_text = float(min(0.99, ax_bbox.y1 + 0.02 + n*0.06)) # stacked upward
    return x_text, y_text

def _entry_vline(ax, x_pos):
    """Optional faint vertical line from the entry x-position to de-emphasize clutter."""
    try:
        ax.axvline(x_pos, linestyle=":", linewidth=0.8, color="grey", alpha=0.35, zorder=0)
    except Exception:
        pass

def annotate_outside_linebox(ax, ts, px, label: str, color: str, draw_vline: bool = True, marker: str = None):
    """
    Generic outside annotation with a connector line from entry point (ts, px).
    The box is placed in a stacked column at the top-right of the axes (figure coords).
    """
    try:
        x_text, y_text = _next_outside_box_position(ax)
        if draw_vline:
            _entry_vline(ax, ts)
        # entry marker (optional)
        if marker is not None:
            ax.scatter([ts], [px], marker=marker, s=80, c=color, edgecolors="none", zorder=1001)
        else:
            ax.scatter([ts], [px], s=60, c=color, zorder=1001)

        ax.annotate(
            label,
            xy=(ts, px),
            xycoords='data',
            xytext=(x_text, y_text),
            textcoords=ax.figure.transFigure,
            ha='right',
            va='bottom',
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, alpha=0.98),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
            annotation_clip=False,
            zorder=2000
        )
    except Exception:
        # fallback to inside minimal text (rare)
        ax.text(ts, px, label, color=color, fontsize=9, fontweight="bold", zorder=2000)

# --- Annotation helpers (updated to use outside stack where applicable) ---
def annotate_signal_box(ax, ts, px, side: str, note: str = "", ypad_frac: float = 0.045):
    # (Kept for internal uses where inside label is still desired)
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
        ax.scatter([ts], [px], s=60, c=("tab:green" if side=="BUY" else "tab:red"), zorder=10)
    except Exception:
        ax.text(ts, px, f" {text}", color=("tab:green" if side=="BUY" else "tab:red"),
                fontsize=10, fontweight="bold")

def annotate_band_rev_outside(ax, ts, px, side: str, note: str = "Band REV"):
    label = f"{'▲ BUY' if side=='BUY' else '▼ SELL'} {note}"
    color = "tab:green" if side == "BUY" else "tab:red"
    annotate_outside_linebox(ax, ts, px, label, color, draw_vline=True, marker="^" if side=="BUY" else "v")

# --- NEW: Outside marker for confirmed reversals ("Price Reversed") ---
def annotate_price_reversed_outside(ax, ts, px, side: str = None):
    color = "tab:green" if (isinstance(side, str) and side.upper() == "BUY") else ("tab:red" if (isinstance(side, str) and side.upper() == "SELL") else "black")
    annotate_outside_linebox(ax, ts, px, "Price Reversed", color, draw_vline=True, marker="*")

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
        return {"time": t, "price": float(p.loc[t]), "kind": "trough"}  # BUY
    if trend_slope < 0 and down_cross.any():
        t = down_cross[down_cross].index[-1]
        return {"time": t, "price": float(p.loc[t]), "kind": "peak"}    # SELL
    return None

# --- NEW: Plain HMA-cross signal (latest) for callout (Daily & Hourly helper) ---
def last_hma_cross_signal(price: pd.Series,
                          hma: pd.Series,
                          trend_slope: float,
                          lookback: int = 60):
    """
    Returns latest {"time","price","side","note"} where price crosses its HMA and agrees with trend_slope.
    """
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    p = _coerce_1d_series(price).astype(float).dropna()
    h = _coerce_1d_series(hma).reindex(p.index).astype(float)
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
        return {"time": t, "price": float(p.loc[t]), "side": "BUY", "note": "HMA55 Cross"}
    if trend_slope < 0 and down_cross.any():
        t = down_cross[down_cross].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL", "note": "HMA55 Cross"}
    return None

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

def _rel_near(a: float, b: float, tol: float) -> bool:
    try:
        a = float(a); b = float(b)
    except Exception:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom <= float(tol)

def daily_sr_99_reversal_signal(price: pd.Series,
                                support: pd.Series,
                                resistance: pd.Series,
                                upper99: pd.Series,
                                lower99: pd.Series,
                                trend_slope: float,
                                prox: float = 0.0025,
                                confirm_bars: int = 2):
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(3, confirm_bars + 1) or not np.isfinite(trend_slope) or trend_slope == 0:
        return None

    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    res = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()
    up99 = _coerce_1d_series(upper99).reindex(p.index)
    lo99 = _coerce_1d_series(lower99).reindex(p.index)

    if sup.dropna().empty or res.dropna().empty or up99.dropna().empty or lo99.dropna().empty:
        return None
    if len(up99) < 2 or len(lo99) < 2:
        return None

    def _inc_ok(series: pd.Series, n: int) -> bool:
        s = _coerce_1d_series(series).dropna()
        if len(s) < n+1: return False
        d = np.diff(s.iloc[-(n+1):])
        return bool(np.all(d > 0))

    def _dec_ok(series: pd.Series, n: int) -> bool:
        s = _coerce_1d_series(series).dropna()
        if len(s) < n+1: return False
        d = np.diff(s.iloc[-(n+1):])
        return bool(np.all(d < 0))

    t0 = p.index[-1]
    c0, c1 = float(p.iloc[-1]), float(p.iloc[-2])
    s0, s1 = float(sup.iloc[-1]), float(sup.iloc[-2])
    r0, r1 = float(res.iloc[-1]), float(res.iloc[-2])
    u1 = float(up99.iloc[-2]) if np.isfinite(up99.iloc[-2]) else np.nan
    l1 = float(lo99.iloc[-2]) if np.isfinite(lo99.iloc[-2]) else np.nan

    if trend_slope > 0:
        prev_near_support = (c1 <= s1 * (1.0 + prox))
        support_near_99   = _rel_near(s1, l1, prox)
        going_up          = _inc_ok(p, confirm_bars)
        if prev_near_support and support_near_99 and going_up:
            return {"time": t0, "price": c0, "side": "BUY", "note": "ALERT 99% SR REV"}
    else:
        prev_near_resistance = (c1 >= r1 * (1.0 - prox))
        resistance_near_99   = _rel_near(r1, u1, prox)
        going_down           = _dec_ok(p, confirm_bars)
        if prev_near_resistance and resistance_near_99 and going_down:
            return {"time": t0, "price": c0, "side": "SELL", "note": "ALERT 99% SR REV"}
    return None

# --- NEW: Daily support-touch + confirmed-up scanner helper ---
def find_support_touch_confirmed_up(price: pd.Series,
                                    support: pd.Series,
                                    prox: float = 0.0025,
                                    confirm_bars: int = 2,
                                    lookback_bars: int = 30):
    p = _coerce_1d_series(price).dropna()
    s = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    if p.shape[0] < confirm_bars + 2 or s.dropna().empty:
        return None

    tail = p.iloc[-(lookback_bars + confirm_bars + 1):]
    for t in reversed(tail.iloc[:-confirm_bars].index.tolist()):
        try:
            pc = float(p.loc[t]); sc = float(s.loc[t])
        except Exception:
            continue
        if not (np.isfinite(pc) and np.isfinite(sc)):
            continue
        touched = (pc <= sc * (1.0 + prox))
        if not touched:
            continue
        if _after_all_increasing(p, t, confirm_bars):
            now_close = float(p.iloc[-1])
            gain_pct = (now_close - pc) / pc if pc != 0 else np.nan
            try:
                loc = p.index.get_loc(t)
                if isinstance(loc, slice):
                    loc = loc.start
                bars_since = int((len(p) - 1) - loc)
            except Exception:
                bars_since = np.nan
            return {
                "t_touch": t,
                "touch_close": pc,
                "support": sc,
                "now_close": now_close,
                "gain_pct": gain_pct,
                "bars_since_touch": bars_since
            }
    return None

# --- Sessions & News ---
NY_TZ   = pytz.timezone("America/New_York")
LDN_TZ  = pytz.timezone("Europe/London")

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
        if lo <= dt_open_pst  <= hi: opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi: closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close   = session_markers_for_index(idx, NY_TZ,  8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close,
            "ny_open": ny_open,   "ny_close": ny_close}

def draw_session_lines(ax, lines: dict):
    ax.plot([], [], linestyle="-",  color="tab:blue",   label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue",   label="London Close (PST)")
    ax.plot([], [], linestyle="-",  color="tab:orange", label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")
    for t in lines.get("ldn_open", []):  ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ldn_close", []): ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ny_open", []):   ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:orange", alpha=0.35)
    for t in lines.get("ny_close", []):  ax.axvline(t, linestyle="--", color="tab:orange", linewidth=1.0, alpha=0.35)
    ax.text(0.99, 0.98, "Session times in PST", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7))

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
            "title": item.get("title",""),
            "publisher": item.get("publisher",""),
            "link": item.get("link","")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")

# --- Market-time compressed axis utilities (NEW) ---
def make_market_time_formatter(index: pd.DatetimeIndex) -> FuncFormatter:
    def _fmt(x, _pos=None):
        i = int(round(x))
        if 0 <= i < len(index):
            ts = index[i]
            return ts.strftime("%m-%d %H:%M")
        return ""
    return FuncFormatter(_fmt)

def map_times_to_positions(index: pd.DatetimeIndex, times: list):
    pos = []
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return pos
    for t in times:
        try:
            j = index.get_indexer([pd.Timestamp(t).tz_convert(index.tz)], method="nearest")[0]
        except Exception:
            j = index.get_indexer([pd.Timestamp(t)], method="nearest")[0]
        if j != -1:
            pos.append(j)
    return pos

def map_session_lines_to_positions(lines: dict, index: pd.DatetimeIndex):
    return {k: map_times_to_positions(index, v) for k, v in lines.items()}

def market_time_axis(ax, index: pd.DatetimeIndex):
    ax.set_xlim(0, max(0, len(index) - 1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(make_market_time_formatter(index))

# --- NEW: Robust aligner to ensure PSAR always renders against the plotted index ---
def _align_series_to_index(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    s = _coerce_1d_series(s)
    if s.empty:
        return pd.Series(index=idx, dtype=float)
    out = s.reindex(idx)
    try:
        out = out.interpolate(method="time").ffill().bfill()
    except Exception:
        out = out.ffill().bfill()
    return out.reindex(idx)

# --- NEW (Hourly): HMA(55) cross AFTER S/R reversal signal detection ---
def hma_cross_after_sr_reversal_signal(price: pd.Series,
                                       hma: pd.Series,
                                       support: pd.Series,
                                       resistance: pd.Series,
                                       global_slope: float,
                                       prox: float = 0.0025,
                                       confirm_bars: int = 2,
                                       lookback_bars: int = 120):
    """
    Returns {"time","price","side","note"} when:
      • BUY  if global_slope>0, price touched/near Support then printed `confirm_bars` higher closes,
             and later crossed ABOVE HMA.
      • SELL if global_slope<0, price touched/near Resistance then printed `confirm_bars` lower closes,
             and later crossed BELOW HMA.
    """
    if not np.isfinite(global_slope) or global_slope == 0:
        return None

    p = _coerce_1d_series(price).dropna().astype(float)
    h = _coerce_1d_series(hma).reindex(p.index).astype(float)
    s = _coerce_1d_series(support).reindex(p.index).ffill().bfill().astype(float)
    r = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill().astype(float)

    if p.empty or h.dropna().empty or s.dropna().empty or r.dropna().empty:
        return None

    start = max(0, len(p) - int(max(lookback_bars, confirm_bars + 3)))

    if global_slope > 0:
        # scan for latest support touch then up-confirm then HMA up-cross
        for i in range(len(p) - confirm_bars - 1, start - 1, -1):
            if not (np.isfinite(p.iloc[i]) and np.isfinite(s.iloc[i])):
                continue
            touched_support = p.iloc[i] <= s.iloc[i] * (1.0 + prox)
            if not touched_support:
                continue
            # confirm higher closes
            ok = True
            for k in range(1, confirm_bars + 1):
                if i + k >= len(p) or not (p.iloc[i + k] > p.iloc[i + k - 1]):
                    ok = False; break
            if not ok:
                continue
            # find first up-cross of HMA after confirmation
            t0 = i + confirm_bars
            for t in range(t0 + 1, len(p)):
                if not (np.isfinite(p.iloc[t - 1]) and np.isfinite(p.iloc[t]) and
                        np.isfinite(h.iloc[t - 1]) and np.isfinite(h.iloc[t])):
                    continue
                up_cross = (p.iloc[t - 1] < h.iloc[t - 1]) and (p.iloc[t] >= h.iloc[t])
                heading_up = (p.iloc[t] > p.iloc[t - 1]) and (p.iloc[t - 1] >= p.iloc[t - 2] if t >= 2 else True)
                if up_cross and heading_up:
                    return {"time": p.index[t], "price": float(p.iloc[t]),
                            "side": "BUY", "note": "HMA55 after Support REV"}
        return None

    else:
        # scan for latest resistance touch then down-confirm then HMA down-cross
        for i in range(len(p) - confirm_bars - 1, start - 1, -1):
            if not (np.isfinite(p.iloc[i]) and np.isfinite(r.iloc[i])):
                continue
            touched_res = p.iloc[i] >= r.iloc[i] * (1.0 - prox)
            if not touched_res:
                continue
            ok = True
            for k in range(1, confirm_bars + 1):
                if i + k >= len(p) or not (p.iloc[i + k] < p.iloc[i + k - 1]):
                    ok = False; break
            if not ok:
                continue
            t0 = i + confirm_bars
            for t in range(t0 + 1, len(p)):
                if not (np.isfinite(p.iloc[t - 1]) and np.isfinite(p.iloc[t]) and
                        np.isfinite(h.iloc[t - 1]) and np.isfinite(h.iloc[t])):
                    continue
                down_cross = (p.iloc[t - 1] > h.iloc[t - 1]) and (p.iloc[t] <= h.iloc[t])
                heading_down = (p.iloc[t] < p.iloc[t - 1]) and (p.iloc[t - 1] <= p.iloc[t - 2] if t >= 2 else True)
                if down_cross and heading_down:
                    return {"time": p.index[t], "price": float(p.iloc[t]),
                            "side": "SELL", "note": "HMA55 after Resistance REV"}
        return None

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10

# Tabs (ADDED new "HMA 55 Reversals")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Upward Slope Stickers",
    "Daily Support Reversals",
    "HMA 55 Reversals"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")
    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly lookback:", ["24h", "48h", "96h"],
                              index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
                              key="hour_range_select")
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    auto_run = st.session_state.run_all

    if st.button("Run Forecast", key="btn_run_forecast") or auto_run:
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
        st.session_state.update({
            "df_hist": df_hist, "df_ohlc": df_ohlc,
            "fc_idx": fc_idx, "fc_vals": fc_vals, "fc_ci": fc_ci,
            "intraday": intraday, "ticker": sel, "chart": chart,
            "hour_range": hour_range, "run_all": True
        })

    st.caption("The Slope Line serves as an informational tool that signals potential trend changes and should be used for risk management rather than trading decisions. Trading based on the slope should only occur when it aligns with the trend line.")

    caution_below_btn = st.empty()

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)
        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        fx_news = fetch_yf_news(sel, window_days=news_window_days) if (mode == "Forex" and show_fx_news) else pd.DataFrame()

        # ----- Daily (Price only) -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()

            # Trendline with 99% (~2.576σ) band and R² (Daily ONLY)
            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily, z=Z_FOR_99)

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, _, _ = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            # --- Daily PSAR (purple) ---
            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()

            # map to the selected daily view
            df_show     = subset_by_daily_view(df, daily_view)
            psar_d_show = _align_series_to_index(psar_d_df["PSAR"], df_show.index) if (show_psar and not psar_d_df.empty and "PSAR" in psar_d_df) else pd.Series(index=df_show.index, dtype=float)

            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show  = bb_up_d.reindex(df_show.index)
            bb_lo_d_show  = bb_lo_d.reindex(df_show.index)

            # HMA lines (for star-cross signals too)
            hma_d_full = compute_hma(df, period=hma_period).reindex(df_show.index)
            hma_d_sig55 = compute_hma(df, period=55).reindex(df_show.index)  # fixed for signal

            fig, ax = plt.subplots(figsize=(14, 6))
            plt.subplots_adjust(top=0.86, right=0.995, left=0.06)

            ax.set_title(f"{sel} Daily — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="Price", linewidth=1.4)
            ax.plot(ema30_show.index, ema30_show.values, "--", alpha=0.4, linewidth=1.0, label="_nolegend_")

            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.04, label="_nolegend_")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=0.9, alpha=0.35, label="_nolegend_")

            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.2, color="black",
                        alpha=0.55, label="Kijun")

            if show_hma and not hma_d_full.dropna().empty:
                ax.plot(hma_d_full.index, hma_d_full.values, "-", linewidth=1.3, alpha=0.9, label="HMA")

            # --- PLOT PSAR as a purple line (aligned & on top) ---
            if show_psar and not psar_d_show.dropna().empty:
                ax.plot(psar_d_show.index, psar_d_show.values, "-", linewidth=1.8, color="purple", alpha=0.95, label="PSAR", zorder=6)

            if not yhat_d_show.empty:
                slope_col_d = "tab:green" if m_d >= 0 else "tab:red"
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=3.2, color=slope_col_d, label="Trend")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")
                ax.plot(lower_d_show.index, lower_d_show.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="")

            # DAILY — Support/Resistance lines
            try:
                res_val_d = float(res30_show.iloc[-1])
                sup_val_d = float(sup30_show.iloc[-1])
                if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                    ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:red", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                    ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:green", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                    label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
                    label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")
            except Exception:
                res_val_d = sup_val_d = np.nan

            # --- Signals/badges (Daily) ---
            badges_top = []

            band_sig_d = last_band_reversal_signal(
                price=df_show, band_upper=upper_d_show, band_lower=lower_d_show,
                trend_slope=m_d, prox=sr_prox_pct, confirm_bars=rev_bars_confirm
            )
            if band_sig_d is not None and band_sig_d.get("side") == "BUY":
                badges_top.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_d['price'])}", "tab:green"))
                annotate_buy_triangle(ax, band_sig_d["time"], band_sig_d["price"])
            elif band_sig_d is not None and band_sig_d.get("side") == "SELL":
                annotate_band_rev_outside(ax, band_sig_d["time"], band_sig_d["price"], band_sig_d["side"], note=band_sig_d.get("note",""))

            # Star (Daily)
            star_d = last_reversal_star(df_show, trend_slope=m_d, lookback=20, confirm_bars=rev_bars_confirm)
            if star_d is not None:
                annotate_star(ax, star_d["time"], star_d["price"], star_d["kind"], show_text=False)
                if star_d.get("kind") == "trough":
                    badges_top.append((f"★ Trough REV @{fmt_price_val(star_d['price'])}", "tab:green"))
                elif star_d.get("kind") == "peak":
                    badges_top.append((f"★ Peak REV @{fmt_price_val(star_d['price'])}", "tab:red"))

            # HMA-cross star (Daily) — BUY badge only (no in-chart star), SELL has blue star + badge
            hma_cross_star = last_hma_cross_star(df_show, hma_d_full, trend_slope=m_d, lookback=30)
            if hma_cross_star is not None:
                if hma_cross_star["kind"] == "trough":
                    badges_top.append((f"★ Buy HMA Cross @{fmt_price_val(hma_cross_star['price'])}", "black"))
                else:
                    annotate_star(ax, hma_cross_star["time"], hma_cross_star["price"], hma_cross_star["kind"], show_text=False, color_override="tab:blue")
                    badges_top.append((f"★ Sell HMA Cross @{fmt_price_val(hma_cross_star['price'])}", "tab:blue"))

            # NEW: Daily HMA(55) outside callout (de-cluttered)
            hma_callout_d = last_hma_cross_signal(df_show, hma_d_sig55, trend_slope=m_d, lookback=60)
            if hma_callout_d is not None:
                color = "tab:green" if hma_callout_d["side"] == "BUY" else "tab:red"
                label = f"{'▲ BUY' if hma_callout_d['side']=='BUY' else '▼ SELL'} {hma_callout_d['note']}"
                annotate_outside_linebox(ax, hma_callout_d["time"], hma_callout_d["price"], label, color, draw_vline=True)

            # Breakout (Daily)
            breakout_d = last_breakout_signal(
                price=df_show, resistance=res30_show, support=sup30_show,
                prox=sr_prox_pct, confirm_bars=rev_bars_confirm
            )
            if breakout_d is not None:
                if breakout_d["dir"] == "UP":
                    annotate_breakout(ax, breakout_d["time"], breakout_d["price"], "UP")
                    badges_top.append((f"▲ BREAKOUT @{fmt_price_val(breakout_d['price'])}", "tab:green"))
                else:
                    annotate_breakout(ax, breakout_d["time"], breakout_d["price"], "DOWN")
                    badges_top.append((f"▼ BREAKDOWN @{fmt_price_val(breakout_d['price'])}", "tab:red"))

            # 99% SR Reversal Alert (Daily)
            sr99_sig = daily_sr_99_reversal_signal(
                price=df_show,
                support=sup30_show,
                resistance=res30_show,
                upper99=upper_d_show,
                lower99=lower_d_show,
                trend_slope=m_d,
                prox=sr_prox_pct,
                confirm_bars=rev_bars_confirm
            )
            if sr99_sig is not None:
                # Keep inside compact box for SR alert
                annotate_signal_box(ax, sr99_sig["time"], sr99_sig["price"], side=sr99_sig["side"], note=sr99_sig["note"])
                if sr99_sig["side"] == "BUY":
                    badges_top.append((f"▲ BUY ALERT 99% SR REV @{fmt_price_val(sr99_sig['price'])}", "tab:green"))
                else:
                    badges_top.append((f"▼ SELL ALERT 99% SR REV @{fmt_price_val(sr99_sig['price'])}", "tab:red"))

            # NEW: Add outside "Price Reversed" marker (Daily)
            if band_sig_d is not None:
                annotate_price_reversed_outside(ax, band_sig_d["time"], band_sig_d["price"], side=band_sig_d.get("side"))
            elif star_d is not None:
                _side = "BUY" if star_d.get("kind") == "trough" else "SELL"
                annotate_price_reversed_outside(ax, star_d["time"], star_d["price"], side=_side)

            draw_top_badges(ax, badges_top)

            # TOP instruction banner (Daily) — use LOCAL=m_d and GLOBAL=m_d (aligned)
            try:
                px_val_d  = float(df_show.iloc[-1])
                confirm_side = sr99_sig["side"] if sr99_sig is not None else None
                draw_instruction_ribbons(ax, m_d, sup_val_d, res_val_d, px_val_d, sel,
                                         confirm_side=confirm_side,
                                         global_slope=m_d)  # aligned on daily
            except Exception:
                pass

            ax.text(0.99, 0.02,
                    f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}  •  Slope: {fmt_slope(m_d)}/bar",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

            _simplify_axes(ax)
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.4)
            pad_right_xaxis(ax, frac=0.06)
            st.pyplot(fig)

        # ----- Hourly (Price only) -----
        if chart in ("Hourly","Both"):
            intraday = st.session_state.intraday
            if intraday is None or intraday.empty or "Close" not in intraday:
                st.warning("No intraday data available.")
            else:
                hc = intraday["Close"].astype(float).ffill()
                xh = np.arange(len(hc), dtype=float)
                if len(hc.dropna()) >= 2:
                    try:
                        coef = np.polyfit(xh, hc.values.astype(float), 1)
                        slope_h = float(coef[0]); intercept_h = float(coef[1])
                        trend_h = slope_h * xh + intercept_h
                    except Exception:
                        slope_h = 0.0
                        intercept_h = float(hc.iloc[-1]) if len(hc) else 0.0
                        trend_h = np.full_like(xh, intercept_h, dtype=float)
                else:
                    slope_h = 0.0
                    intercept_h = float(hc.iloc[-1]) if len(hc) else 0.0
                    trend_h = np.full_like(xh, intercept_h, dtype=float)

                he = hc.ewm(span=20).mean()
                res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
                sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()
                st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday else pd.Series(index=hc.index, dtype=float)

                kijun_h = pd.Series(index=hc.index, dtype=float)
                if {'High','Low','Close'}.issubset(intraday.columns) and show_ichi:
                    _, kijun_h, _, _, _ = ichimoku_lines(intraday["High"], intraday["Low"], intraday["Close"],
                                                         conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                    kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

                bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
                # HMA plotted per user setting, but signals use fixed HMA(55)
                hma_h_plot = compute_hma(hc, period=hma_period)
                hma_h_sig  = compute_hma(hc, period=55)

                psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
                psar_h_aligned = _align_series_to_index(psar_h_df["PSAR"], hc.index) if (show_psar and not psar_h_df.empty and "PSAR" in psar_h_df) else pd.Series(index=hc.index, dtype=float)

                # Hourly regression slope & bands (GLOBAL)
                yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)

                # Show caution banner below button if LOCAL vs GLOBAL disagree
                try:
                    if np.isfinite(slope_h) and np.isfinite(m_h) and (slope_h * m_h < 0):
                        caution_below_btn.warning("ALERT: Please exercise caution while trading at this moment, as the current slope indicates that the dash trendline may be reversing. A reversal occurs near the 100% or 0% Fibonacci retracement levels. Once the reversal is confirmed, the trendline changes direction")
                except Exception:
                    pass

                idx_mt = hc.index
                x_mt = np.arange(len(idx_mt), dtype=float)

                def _pos(ts):
                    ix = idx_mt.get_indexer([ts], method="nearest")[0]
                    return float(ix) if ix != -1 else np.nan

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.86, right=0.995, left=0.06)

                trend_color = "tab:green" if slope_h >= 0 else "tab:red"
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")

                ax2.plot(x_mt, hc.values, label="Price", linewidth=1.2)
                ax2.plot(x_mt, he.reindex(idx_mt).values, "--", alpha=0.45, linewidth=0.9, label="_nolegend_")
                ax2.plot(x_mt, trend_h, "--", label="Trend", linewidth=2.4, color=trend_color, alpha=0.95)

                if show_hma and not hma_h_plot.dropna().empty:
                    ax2.plot(x_mt, hma_h_plot.reindex(idx_mt).values, "-", linewidth=1.3, alpha=0.9, label="HMA")

                if show_ichi and not kijun_h.dropna().empty:
                    ax2.plot(x_mt, kijun_h.reindex(idx_mt).values, "-", linewidth=1.1, color="black", alpha=0.55, label="Kijun")

                if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
                    ax2.fill_between(x_mt, bb_lo_h.reindex(idx_mt).values, bb_up_h.reindex(idx_mt).values, alpha=0.04, label="_nolegend_")
                    ax2.plot(x_mt, bb_mid_h.reindex(idx_mt).values, "-", linewidth=0.8, alpha=0.3, label="_nolegend_")

                if show_psar and not psar_h_aligned.dropna().empty:
                    ax2.plot(x_mt, psar_h_aligned.reindex(idx_mt).values, "-", linewidth=1.8, color="purple", alpha=0.95, label="PSAR", zorder=6)

                res_val = sup_val = px_val = np.nan
                try:
                    res_val = float(res_h.iloc[-1]); sup_val = float(sup_h.iloc[-1]); px_val = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=0, xmax=len(x_mt)-1, colors="tab:red",   linestyles="-", linewidth=2.0, alpha=0.95, label="_nolegend_")
                    ax2.hlines(sup_val, xmin=0, xmax=len(x_mt)-1, colors="tab:green", linestyles="-", linewidth=2.0, alpha=0.95, label="_nolegend_")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                badges_top_h = []

                band_sig_h = last_band_reversal_signal(
                    price=hc, band_upper=upper_h, band_lower=lower_h,
                    trend_slope=m_h, prox=sr_prox_pct, confirm_bars=rev_bars_confirm
                )
                if band_sig_h is not None:
                    tpos = _pos(band_sig_h["time"])
                    if np.isfinite(tpos):
                        if band_sig_h.get("side") == "BUY":
                            badges_top_h.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_h['price'])}", "tab:green"))
                            annotate_buy_triangle(ax2, tpos, band_sig_h["price"])
                        else:
                            annotate_band_rev_outside(ax2, tpos, band_sig_h["price"], band_sig_h["side"], note=band_sig_h.get("note",""))

                star_h = last_reversal_star(hc, trend_slope=m_h, lookback=20, confirm_bars=rev_bars_confirm)
                if star_h is not None:
                    tpos_star = _pos(star_h["time"])
                    if np.isfinite(tpos_star):
                        annotate_star(ax2, tpos_star, star_h["price"], star_h["kind"], show_text=False)
                        if star_h.get("kind") == "trough":
                            badges_top_h.append((f"★ Trough REV @{fmt_price_val(star_h['price'])}", "tab:green"))
                        elif star_h.get("kind") == "peak":
                            badges_top_h.append((f"★ Peak REV @{fmt_price_val(star_h['price'])}", "tab:red"))

                breakout_h = last_breakout_signal(
                    price=hc, resistance=res_h, support=sup_h,
                    prox=sr_prox_pct, confirm_bars=rev_bars_confirm
                )
                if breakout_h is not None:
                    tpos_bo = _pos(breakout_h["time"])
                    if np.isfinite(tpos_bo):
                        if breakout_h["dir"] == "UP":
                            annotate_breakout(ax2, tpos_bo, breakout_h["price"], "UP")
                            badges_top_h.append((f"▲ BREAKOUT @{fmt_price_val(breakout_h['price'])}", "tab:green"))
                        else:
                            annotate_breakout(ax2, tpos_bo, breakout_h["price"], "DOWN")
                            badges_top_h.append((f"▼ BREAKDOWN @{fmt_price_val(breakout_h['price'])}", "tab:red"))

                # NEW: HMA(55) cross AFTER S/R reversal signal (uses GLOBAL slope m_h) — OUTSIDE CALLOUT
                hma_sr_sig = hma_cross_after_sr_reversal_signal(
                    price=hc,
                    hma=hma_h_sig,  # fixed 55 for signal
                    support=sup_h,
                    resistance=res_h,
                    global_slope=m_h,
                    prox=sr_prox_pct,
                    confirm_bars=rev_bars_confirm,
                    lookback_bars=slope_lb_hourly
                )
                if hma_sr_sig is not None:
                    tpos_sig = _pos(hma_sr_sig["time"])
                    if np.isfinite(tpos_sig):
                        color = "tab:green" if hma_sr_sig["side"] == "BUY" else "tab:red"
                        label = f"{'▲ BUY' if hma_sr_sig['side']=='BUY' else '▼ SELL'} {hma_sr_sig['note']}"
                        # Outside callout with connector (declutter)
                        annotate_outside_linebox(ax2, tpos_sig, hma_sr_sig["price"], label, c_
