# =========================
# Part 1/10 — bullbear.py  (UPDATED: Ribbon Tabs + Beautiful Chart Styling)
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
     UPDATED (THIS REQUEST):
     Sidebar — world-class UI/UX (STYLE ONLY)
     Scope strictly to sidebar to avoid affecting main app.
     ========================= */
  section[data-testid="stSidebar"]{
    background:
      radial-gradient(1200px 900px at 0% 0%, rgba(99,102,241,0.20), transparent 55%),
      radial-gradient(900px 650px at 0% 65%, rgba(59,130,246,0.18), transparent 52%),
      linear-gradient(180deg, rgba(15,23,42,1) 0%, rgba(17,24,39,1) 100%);
    border-right: 1px solid rgba(255,255,255,0.10);
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
    padding: 0.95rem 0.85rem 1.2rem 0.85rem;
  }

  /* Typography in sidebar */
  section[data-testid="stSidebar"] *{
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] h4{
    color: rgba(255,255,255,0.94) !important;
    letter-spacing: 0.01em;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] li,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div{
    color: rgba(229,231,235,0.88);
  }

  /* Sidebar header card */
  section[data-testid="stSidebar"] .sb-brand{
    padding: 0.85rem 0.85rem 0.75rem 0.85rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 26px rgba(0,0,0,0.22);
    margin-bottom: 0.65rem;
  }
  section[data-testid="stSidebar"] .sb-brand-row{
    display: flex;
    align-items: center;
    gap: 0.65rem;
  }
  section[data-testid="stSidebar"] .sb-brand-icon{
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.28);
    box-shadow: 0 10px 20px rgba(0,0,0,0.16);
    font-size: 18px;
  }
  section[data-testid="stSidebar"] .sb-brand-title{
    font-weight: 900;
    font-size: 1.05rem;
    color: rgba(255,255,255,0.96);
    line-height: 1.1;
  }
  section[data-testid="stSidebar"] .sb-brand-sub{
    margin-top: 0.18rem;
    font-size: 0.90rem;
    color: rgba(229,231,235,0.86);
    line-height: 1.1;
  }
  section[data-testid="stSidebar"] .sb-pill{
    display: inline-block;
    padding: 0.16rem 0.55rem;
    border-radius: 999px;
    background: rgba(59,130,246,0.16);
    border: 1px solid rgba(59,130,246,0.26);
    color: rgba(255,255,255,0.92);
    font-weight: 800;
  }

  /* Auto-refresh status card */
  section[data-testid="stSidebar"] .sb-status{
    padding: 0.75rem 0.85rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 24px rgba(0,0,0,0.20);
    margin: 0.65rem 0 0.75rem 0;
  }
  section[data-testid="stSidebar"] .sb-status-title{
    font-weight: 900;
    color: rgba(255,255,255,0.95);
    font-size: 0.95rem;
    margin-bottom: 0.45rem;
    letter-spacing: 0.01em;
  }
  section[data-testid="stSidebar"] .sb-status-grid{
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.35rem;
  }
  section[data-testid="stSidebar"] .sb-status-row{
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.35rem 0.55rem;
    border-radius: 12px;
    background: rgba(0,0,0,0.18);
    border: 1px solid rgba(255,255,255,0.08);
  }
  section[data-testid="stSidebar"] .sb-k{
    font-size: 0.84rem;
    color: rgba(229,231,235,0.78);
  }
  section[data-testid="stSidebar"] .sb-v{
    font-size: 0.84rem;
    color: rgba(255,255,255,0.92);
    font-weight: 800;
    text-align: right;
    white-space: nowrap;
  }

  /* Section headers (subheader) */
  section[data-testid="stSidebar"] h3{
    margin-top: 0.85rem !important;
    margin-bottom: 0.40rem !important;
    font-size: 0.90rem !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: rgba(255,255,255,0.82) !important;
  }

  /* Widgets: labels */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stMarkdown label{
    color: rgba(229,231,235,0.86) !important;
    font-weight: 700 !important;
  }

  /* Widgets: Selectbox / multiselect / date input (BaseWeb) */
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 22px rgba(0,0,0,0.18) !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="select"] *{
    color: rgba(255,255,255,0.92) !important;
  }

  /* Widgets: Slider */
  section[data-testid="stSidebar"] div[data-baseweb="slider"]{
    padding: 0.15rem 0.25rem 0.10rem 0.25rem !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
  }
  section[data-testid="stSidebar"] div[data-baseweb="slider"] *{
    color: rgba(255,255,255,0.92) !important;
  }

  /* Widgets: Checkbox */
  section[data-testid="stSidebar"] label[data-baseweb="checkbox"]{
    padding: 0.30rem 0.40rem !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    transition: transform 120ms ease, background 120ms ease, border-color 120ms ease !important;
  }
  section[data-testid="stSidebar"] label[data-baseweb="checkbox"]:hover{
    transform: translateY(-1px) !important;
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.14) !important;
  }

  /* Buttons in sidebar */
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button{
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.92) !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 12px 26px rgba(0,0,0,0.20) !important;
    transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease, border-color 120ms ease !important;
  }
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover{
    transform: translateY(-1px) !important;
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.20) !important;
    box-shadow: 0 18px 34px rgba(0,0,0,0.28) !important;
  }
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus{
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.35) !important;
  }

  /* Divider helper */
  section[data-testid="stSidebar"] .sb-divider{
    height: 1px;
    width: 100%;
    margin: 0.70rem 0 0.55rem 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent);
  }

  /* =========================
     Existing styling in file (UNCHANGED):
     (1) Beautiful rectangular ribbon tabs (BaseWeb tabs)
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
     Existing styling in file (UNCHANGED):
     (2) Beautiful chart container styling (Streamlit React UI wrappers)
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

  /* Mobile: keep sidebar usable (stable selectors) */
  @media (max-width: 600px) {
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
      padding: 0.75rem 0.70rem 1.0rem 0.70rem;
    }
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

# UPDATED (THIS REQUEST): sidebar status card (STYLE ONLY — same info)
st.sidebar.markdown(
    f"""
    <div class="sb-status">
      <div class="sb-status-title">⏱️ Auto-refresh status</div>
      <div class="sb-status-grid">
        <div class="sb-status-row"><div class="sb-k">Every</div><div class="sb-v">{REFRESH_INTERVAL//60} min</div></div>
        <div class="sb-status-row"><div class="sb-k">Last refresh</div><div class="sb-v">{pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST</div></div>
        <div class="sb-status-row"><div class="sb-k">Next in</div><div class="sb-v">~{remaining}s</div></div>
      </div>
    </div>
    <div class="sb-divider"></div>
    """,
    unsafe_allow_html=True
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

FIB_ALERT_TEXT = "ALERT: Fibonacci Guidance — Prices often reverse at the 100% and 0% lines. It's essential to implement risk management when trading near these Fibonacci levels."

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
# Part 2/10 — bullbear.py
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
# UPDATED (THIS REQUEST): branded sidebar header (STYLE ONLY)
st.sidebar.markdown(
    f"""
    <div class="sb-brand">
      <div class="sb-brand-row">
        <div class="sb-brand-icon">⚙️</div>
        <div>
          <div class="sb-brand-title">Configuration</div>
          <div class="sb-brand-sub">Asset Class: <span class="sb-pill">{mode}</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

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

show_fibs = st.sidebar.checkbox("Show Fibonacci", value=True, key="sb_show_fibs")

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
        "HKDJPY=X","USDCAD=X","USDCNY=X","EURGBP=X","EURCAD=X","NZDJPY=X","USDKRW=X",
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

def fib_reversal_trigger_from_extremes(series_like,
                                      proximity_pct_of_range: float = 0.02,
                                      confirm_bars: int = 2,
                                      lookback_bars: int = 60):
    """
    CONFIRMED BUY:
      - price touched near Fib 100% (low) within lookback
      - then prints `confirm_bars` consecutive higher closes (reversal up from low)
    CONFIRMED SELL:
      - price touched near Fib 0% (high) within lookback
      - then prints `confirm_bars` consecutive lower closes (reversal down from high)

    Returns dict or None.
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.empty or len(s) < max(4, int(confirm_bars) + 2):
        return None

    lb = max(10, int(lookback_bars))
    s = s.iloc[-lb:] if len(s) > lb else s

    fibs = fibonacci_levels(s)
    if not fibs:
        return None

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        return None

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return None

    near_hi = s >= (hi - tol)
    near_lo = s <= (lo + tol)

    last_hi_touch = near_hi[near_hi].index[-1] if near_hi.any() else None
    last_lo_touch = near_lo[near_lo].index[-1] if near_lo.any() else None

    def _confirmed_up(from_time):
        seg = s.loc[from_time:]
        return bool(len(seg) >= int(confirm_bars) + 1 and np.all(np.diff(seg.iloc[-(int(confirm_bars)+1):]) > 0))

    def _confirmed_down(from_time):
        seg = s.loc[from_time:]
        return bool(len(seg) >= int(confirm_bars) + 1 and np.all(np.diff(seg.iloc[-(int(confirm_bars)+1):]) < 0))

    buy_tr = None
    if last_lo_touch is not None and _confirmed_up(last_lo_touch):
        buy_tr = {
            "side": "BUY",
            "from_level": "100%",
            "touch_time": last_lo_touch,
            "touch_price": float(s.loc[last_lo_touch]) if np.isfinite(s.loc[last_lo_touch]) else np.nan,
            "last_time": s.index[-1],
            "last_price": float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan
        }

    sell_tr = None
    if last_hi_touch is not None and _confirmed_down(last_hi_touch):
        sell_tr = {
            "side": "SELL",
            "from_level": "0%",
            "touch_time": last_hi_touch,
            "touch_price": float(s.loc[last_hi_touch]) if np.isfinite(s.loc[last_hi_touch]) else np.nan,
            "last_time": s.index[-1],
            "last_price": float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan
        }

    if buy_tr is None and sell_tr is None:
        return None
    if buy_tr is None:
        return sell_tr
    if sell_tr is None:
        return buy_tr
    return buy_tr if buy_tr["touch_time"] >= sell_tr["touch_time"] else sell_tr

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
# Part 3/10 — bullbear.py
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

# ---------------------------
# NEW (THIS REQUEST): Fibonacci Buy/Sell markers (Price chart area)
# ---------------------------
def overlay_fib_npx_signals(ax,
                            price: pd.Series,
                            buy_mask: pd.Series,
                            sell_mask: pd.Series,
                            label_buy: str = "Fibonacci BUY",
                            label_sell: str = "Fibonacci SELL"):
    """
    Plot Fibonacci BUY/SELL markers on the PRICE chart.

    Uses buy_mask/sell_mask computed from:
      - price near Fib 100% (low) / 0% (high)
      - NPX crosses 0.0 upward/downward (recent)
    """
    p = _coerce_1d_series(price)
    bm = _coerce_1d_series(buy_mask).reindex(p.index).fillna(0).astype(bool) if buy_mask is not None else pd.Series(False, index=p.index)
    sm = _coerce_1d_series(sell_mask).reindex(p.index).fillna(0).astype(bool) if sell_mask is not None else pd.Series(False, index=p.index)

    buy_idx = list(bm[bm].index)
    sell_idx = list(sm[sm].index)

    if buy_idx:
        ax.scatter(buy_idx, p.loc[buy_idx], marker="^", s=120, color="tab:green", zorder=11, label=label_buy)
        for t in buy_idx:
            try:
                ax.text(t, float(p.loc[t]), "  FIB BUY", va="bottom", fontsize=9, color="tab:green", fontweight="bold", zorder=12)
            except Exception:
                pass

    if sell_idx:
        ax.scatter(sell_idx, p.loc[sell_idx], marker="v", s=120, color="tab:red", zorder=11, label=label_sell)
        for t in sell_idx:
            try:
                ax.text(t, float(p.loc[t]), "  FIB SELL", va="top", fontsize=9, color="tab:red", fontweight="bold", zorder=12)
            except Exception:
                pass

# ---------------------------
# Slope BUY/SELL Trigger (leaderline + legend)
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


# =========================
# Part 4/10 — bullbear.py
# =========================
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

# ---------------------------
# NEW (THIS REQUEST): Fib touch + NPX(0.0) cross logic for Fibonacci BUY/SELL signals
# ---------------------------
def npx_zero_cross_masks(npx: pd.Series, level: float = 0.0):
    """
    NPX cross of a constant level (default 0.0):
      - cross_up: npx goes from < level to >= level
      - cross_dn: npx goes from > level to <= level
    """
    s = _coerce_1d_series(npx)
    prev = s.shift(1)
    cross_up = (s >= float(level)) & (prev < float(level))
    cross_dn = (s <= float(level)) & (prev > float(level))
    return cross_up.fillna(False), cross_dn.fillna(False)

def fib_touch_masks(price: pd.Series, proximity_pct_of_range: float = 0.02):
    """
    Returns (near_hi_0pct, near_lo_100pct, fibs_dict).
    'near' uses a tolerance = proximity_pct_of_range * (fib_range).
    """
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        idx = _coerce_1d_series(price).index
        return (pd.Series(False, index=idx), pd.Series(False, index=idx), {})

    fibs = fibonacci_levels(p)
    if not fibs:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), {})

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(rng)) or rng <= 0:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), fibs)

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), fibs)

    near_hi = (p >= (hi - tol)).reindex(p.index, fill_value=False)
    near_lo = (p <= (lo + tol)).reindex(p.index, fill_value=False)
    return near_hi, near_lo, fibs

def fib_npx_zero_cross_signal_masks(price: pd.Series,
                                   npx: pd.Series,
                                   horizon_bars: int = 15,
                                   proximity_pct_of_range: float = 0.02,
                                   npx_level: float = 0.0):
    """
    Fibonacci BUY mask:
      - NPX crosses UP through 0.0
      - AND price touched near Fib 100% (low) within last `horizon_bars` (including current)

    Fibonacci SELL mask:
      - NPX crosses DOWN through 0.0
      - AND price touched near Fib 0% (high) within last `horizon_bars` (including current)
    """
    p = _coerce_1d_series(price)
    x = _coerce_1d_series(npx).reindex(p.index)

    near_hi, near_lo, fibs = fib_touch_masks(p, proximity_pct_of_range=float(proximity_pct_of_range))
    up0, dn0 = npx_zero_cross_masks(x, level=float(npx_level))

    hz = max(1, int(horizon_bars))
    touched_lo_recent = near_lo.rolling(hz + 1, min_periods=1).max().astype(bool)
    touched_hi_recent = near_hi.rolling(hz + 1, min_periods=1).max().astype(bool)

    buy_mask = up0.reindex(p.index, fill_value=False) & touched_lo_recent.reindex(p.index, fill_value=False)
    sell_mask = dn0.reindex(p.index, fill_value=False) & touched_hi_recent.reindex(p.index, fill_value=False)

    return buy_mask.fillna(False), sell_mask.fillna(False), fibs

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


# =========================
# Part 5/10 — bullbear.py
# =========================
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

def regression_slope_reversal_at_fib_extremes(series_like,
                                              slope_lb: int,
                                              proximity_pct_of_range: float = 0.02,
                                              confirm_bars: int = 2,
                                              lookback_bars: int = 120):
    """
    Returns dict when BOTH are true:
      1) price touched near Fib 0% (high) or 100% (low)
      2) regression slope sign flipped after that touch
         + confirms reversal via consecutive closes
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return None

    lb = int(max(10, lookback_bars))
    s = s.iloc[-lb:] if len(s) > lb else s
    if len(s) < max(6, int(slope_lb) + 3):
        return None

    fibs = fibonacci_levels(s)
    if not fibs:
        return None

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(rng)) or rng <= 0:
        return None

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return None

    near_hi = s >= (hi - tol)
    near_lo = s <= (lo + tol)
    last_hi_touch = near_hi[near_hi].index[-1] if near_hi.any() else None
    last_lo_touch = near_lo[near_lo].index[-1] if near_lo.any() else None

    _, _, _, m_curr, _ = regression_with_band(s, lookback=int(slope_lb))

    def _pre_slope_at(t_touch):
        seg = _coerce_1d_series(s.loc[:t_touch]).dropna().tail(int(slope_lb))
        if len(seg) < 3:
            return np.nan
        _, _, _, m_pre, _ = regression_with_band(seg, lookback=int(slope_lb))
        return float(m_pre) if np.isfinite(m_pre) else np.nan

    buy_rev = None
    if last_lo_touch is not None:
        m_pre = _pre_slope_at(last_lo_touch)
        seg_after = s.loc[last_lo_touch:]
        if np.isfinite(m_pre) and np.isfinite(m_curr):
            if (float(m_pre) < 0.0) and (float(m_curr) > 0.0) and _n_consecutive_increasing(seg_after, int(confirm_bars)):
                buy_rev = {
                    "side": "BUY",
                    "from_level": "100%",
                    "touch_time": last_lo_touch,
                    "touch_price": float(s.loc[last_lo_touch]) if np.isfinite(s.loc[last_lo_touch]) else np.nan,
                    "pre_slope": float(m_pre),
                    "curr_slope": float(m_curr),
                }

    sell_rev = None
    if last_hi_touch is not None:
        m_pre = _pre_slope_at(last_hi_touch)
        seg_after = s.loc[last_hi_touch:]
        if np.isfinite(m_pre) and np.isfinite(m_curr):
            if (float(m_pre) > 0.0) and (float(m_curr) < 0.0) and _n_consecutive_decreasing(seg_after, int(confirm_bars)):
                sell_rev = {
                    "side": "SELL",
                    "from_level": "0%",
                    "touch_time": last_hi_touch,
                    "touch_price": float(s.loc[last_hi_touch]) if np.isfinite(s.loc[last_hi_touch]) else np.nan,
                    "pre_slope": float(m_pre),
                    "curr_slope": float(m_curr),
                }

    if buy_rev is None and sell_rev is None:
        return None
    if buy_rev is None:
        return sell_rev
    if sell_rev is None:
        return buy_rev

    return buy_rev if buy_rev["touch_time"] >= sell_rev["touch_time"] else sell_rev

def annotate_reverse_possible(ax, rev_info: dict, text: str = "Reverse Possible"):
    if not isinstance(rev_info, dict):
        return
    t = rev_info.get("touch_time", None)
    y = rev_info.get("touch_price", np.nan)
    side = str(rev_info.get("side", "")).upper()
    if t is None or (not np.isfinite(y)):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    va = "bottom" if side == "BUY" else "top"
    ax.text(
        t, y,
        f"  {text}",
        color=col,
        fontsize=10,
        fontweight="bold",
        va=va,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.80),
        zorder=25
    )
# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Support / Resistance (rolling)
# ---------------------------
def rolling_support_resistance(close: pd.Series, lookback: int = 60):
    c = _coerce_1d_series(close).astype(float)
    if c.empty:
        idx = c.index
        return (pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float))
    lb = max(5, int(lookback))
    sup = c.rolling(lb, min_periods=max(3, lb // 3)).min()
    res = c.rolling(lb, min_periods=max(3, lb // 3)).max()
    return sup.reindex(c.index), res.reindex(c.index)

def dedup_legend(ax, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), loc=loc)

# ---------------------------
# Session markers (PST)
# ---------------------------
def _session_lines_pst_for_day(day_dt: pd.Timestamp, mode: str = "Forex"):
    """
    Returns list of tuples (ts_pst, label) for vertical session markers on that day.
    """
    if day_dt is None:
        return []
    try:
        day = pd.Timestamp(day_dt).tz_convert(PACIFIC)
    except Exception:
        try:
            day = pd.Timestamp(day_dt).tz_localize(PACIFIC)
        except Exception:
            return []
    d = day.normalize()

    out = []
    if mode == "Forex":
        # London open 08:00 Europe/London, NY open 08:00 US/Eastern
        try:
            london = pytz.timezone("Europe/London")
            ny = pytz.timezone("US/Eastern")
            lon_open = london.localize(datetime(d.year, d.month, d.day, 8, 0, 0))
            ny_open = ny.localize(datetime(d.year, d.month, d.day, 8, 0, 0))
            out.append((lon_open.astimezone(PACIFIC), "London Open"))
            out.append((ny_open.astimezone(PACIFIC), "NY Open"))
        except Exception:
            pass
    else:
        # US stock regular session open 09:30 ET
        try:
            ny = pytz.timezone("US/Eastern")
            ny_open = ny.localize(datetime(d.year, d.month, d.day, 9, 30, 0))
            out.append((ny_open.astimezone(PACIFIC), "US Open"))
        except Exception:
            pass

    return out

# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Ticker selection + run
# ---------------------------
default_symbol = universe[0] if universe else ""

# Protect against stale value not in new universe
if st.session_state.get("ticker") not in universe:
    st.session_state.ticker = default_symbol

ticker = st.selectbox(
    f"Select {'Forex Pair' if mode=='Forex' else 'Stock'}:",
    universe,
    index=universe.index(st.session_state.ticker) if st.session_state.ticker in universe else 0,
    key="sel_ticker_main",
)
st.session_state.ticker = ticker

run_col1, run_col2 = st.columns([1, 3])
if run_col1.button("▶ Run", use_container_width=True, key="btn_run"):
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

run_all = bool(st.session_state.get("run_all", False))

# ---------------------------
# Hour-range selector (intraday)
# ---------------------------
hour_range = st.selectbox(
    "Intraday chart window:",
    ["5 Hours", "1 Day", "5 Days"],
    index=0,
    key="sel_hour_range",
)
st.session_state.hour_range = hour_range

def _intraday_period_for_range(label: str) -> str:
    return {"5 Hours": "1d", "1 Day": "1d", "5 Days": "5d"}.get(label, "1d")

# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Data prep (only when run)
# ---------------------------
if run_all:
    with st.spinner("Fetching data..."):
        # Daily close + OHLC
        df_hist = fetch_hist(ticker)
        df_ohlc = fetch_hist_ohlc(ticker)

        st.session_state.df_hist = df_hist
        st.session_state.df_ohlc = df_ohlc

        # Forecast
        try:
            fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        except Exception:
            fc_idx = pd.DatetimeIndex([], tz=PACIFIC)
            fc_vals = pd.Series(dtype=float)
            fc_ci = pd.DataFrame()
        st.session_state.fc_idx = fc_idx
        st.session_state.fc_vals = fc_vals
        st.session_state.fc_ci = fc_ci

        # Intraday
        intraday_period = _intraday_period_for_range(hour_range)
        intraday = fetch_intraday(ticker, period=intraday_period)
        st.session_state.intraday = intraday

# Pull from session state for display
df_hist = st.session_state.get("df_hist", None)
df_ohlc = st.session_state.get("df_ohlc", None)
fc_idx = st.session_state.get("fc_idx", None)
fc_vals = st.session_state.get("fc_vals", None)
fc_ci = st.session_state.get("fc_ci", None)
intraday = st.session_state.get("intraday", None)

if not run_all:
    st.info("Choose an asset and press **Run**.")
    st.stop()

# ---------------------------
# Derived metrics (daily)
# ---------------------------
hist_view = subset_by_daily_view(df_hist, daily_view) if df_hist is not None else df_hist
ohlc_view = subset_by_daily_view(df_ohlc, daily_view) if df_ohlc is not None else df_ohlc

close_daily = _coerce_1d_series(hist_view).dropna()
close_daily_full = _coerce_1d_series(df_hist).dropna()

last_close = float(close_daily.iloc[-1]) if len(close_daily) else np.nan
bb_ref = _coerce_1d_series(df_hist).dropna()
bb_tail = bb_ref.tail({"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}.get(bb_period, 126))
bb_chg = (bb_tail.iloc[-1] / bb_tail.iloc[0] - 1.0) if len(bb_tail) >= 2 else np.nan

# Global trend slope (max history)
global_yhat, global_slope = slope_line(close_daily_full, lookback=max(120, min(1500, len(close_daily_full))))
global_r2 = regression_r2(close_daily_full, lookback=max(120, min(1500, len(close_daily_full))))

# Daily regression band on selected view
daily_yhat, daily_up, daily_lo, daily_slope, daily_r2 = regression_with_band(close_daily, lookback=int(slope_lb_daily), z=2.0)

# Daily S/R
sup_d, res_d = rolling_support_resistance(close_daily, lookback=sr_lb_daily)

# NTD/NPX (daily)
ntd_d = compute_normalized_trend(close_daily, window=ntd_window) if show_ntd else pd.Series(index=close_daily.index, dtype=float)
npx_d = compute_normalized_price(close_daily, window=ntd_window) if (show_ntd and show_npx_ntd) else pd.Series(index=close_daily.index, dtype=float)

# Fibonacci + Fib/NPX signals (daily)
fib_buy_d = fib_sell_d = None
fibs_d = {}
if show_fibs and len(close_daily) > 5:
    fib_buy_d, fib_sell_d, fibs_d = fib_npx_zero_cross_signal_masks(
        price=close_daily, npx=npx_d, horizon_bars=rev_horizon,
        proximity_pct_of_range=0.02, npx_level=0.0
    )

# Daily BBands + HMA
bb_mid_d = bb_up_d = bb_lo_d = pctb_d = nbb_d = pd.Series(index=close_daily.index, dtype=float)
if show_bbands and len(close_daily) > bb_win:
    bb_mid_d, bb_up_d, bb_lo_d, pctb_d, nbb_d = compute_bbands(close_daily, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

hma_d = compute_hma(close_daily, period=hma_period) if show_hma else pd.Series(index=close_daily.index, dtype=float)

# Daily slope-trigger after band reversal (on daily view)
slope_trigger_daily = None
if len(close_daily) >= max(30, slope_lb_daily) and daily_yhat is not None and not daily_yhat.empty:
    slope_trigger_daily = find_slope_trigger_after_band_reversal(
        price=close_daily, yhat=daily_yhat, upper_band=daily_up, lower_band=daily_lo, horizon=rev_horizon
    )

# Daily fib reversal + regression slope reversal at fib extremes (optional label)
fib_rev_daily = fib_reversal_trigger_from_extremes(
    close_daily, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=min(240, len(close_daily))
) if show_fibs else None

reg_fib_slope_rev_daily = regression_slope_reversal_at_fib_extremes(
    close_daily, slope_lb=slope_lb_daily, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=min(360, len(close_daily))
) if show_fibs else None

# Slope reversal probability (daily)
rev_prob_daily = slope_reversal_probability(
    close_daily, current_slope=daily_slope, hist_window=rev_hist_lb, slope_window=ntd_window, horizon=rev_horizon
)

# Trade instruction uses BOTH global + local slopes (UPDATED earlier)
buy_val = float(sup_d.iloc[-1]) if len(sup_d.dropna()) else np.nan
sell_val = float(res_d.iloc[-1]) if len(res_d.dropna()) else np.nan
trade_text_daily = format_trade_instruction(
    trend_slope=daily_slope,
    buy_val=buy_val,
    sell_val=sell_val,
    close_val=last_close,
    symbol=ticker,
    global_trend_slope=global_slope
)

# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Intraday derived metrics
# ---------------------------
intraday_close = pd.Series(dtype=float)
intraday_ohlc = None
if isinstance(intraday, pd.DataFrame) and not intraday.empty and "Close" in intraday.columns:
    intraday_ohlc = intraday.copy()
    intraday_close = _coerce_1d_series(intraday_ohlc["Close"]).dropna()

    # Apply window choice (5 hours slice)
    if hour_range == "5 Hours":
        t_end = intraday_close.index.max()
        t_start = t_end - pd.Timedelta(hours=5)
        intraday_ohlc = intraday_ohlc.loc[(intraday_ohlc.index >= t_start) & (intraday_ohlc.index <= t_end)]
        intraday_close = _coerce_1d_series(intraday_ohlc["Close"]).dropna()

# Hourly aggregation (from intraday 5m) for indicators in hourly space
hourly_ohlc = None
hourly_close = pd.Series(dtype=float)
if intraday_ohlc is not None and not intraday_ohlc.empty:
    try:
        hourly_ohlc = intraday_ohlc[["Open","High","Low","Close","Volume"]].resample("1H").agg({
            "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
        }).dropna(subset=["Close"])
    except Exception:
        try:
            hourly_ohlc = intraday_ohlc[["Open","High","Low","Close"]].resample("1H").agg({
                "Open":"first","High":"max","Low":"min","Close":"last"
            }).dropna(subset=["Close"])
        except Exception:
            hourly_ohlc = None

if hourly_ohlc is not None and not hourly_ohlc.empty:
    hourly_close = _coerce_1d_series(hourly_ohlc["Close"]).dropna()

# Hourly regression band + S/R
hour_yhat = hour_up = hour_lo = pd.Series(dtype=float)
hour_slope = float("nan")
hour_r2 = float("nan")

sup_h = res_h = pd.Series(dtype=float)
if len(hourly_close) >= 10:
    hour_yhat, hour_up, hour_lo, hour_slope, hour_r2 = regression_with_band(hourly_close, lookback=int(slope_lb_hourly), z=2.0)
    sup_h, res_h = rolling_support_resistance(hourly_close, lookback=sr_lb_hourly)

rev_prob_hourly = slope_reversal_probability(
    hourly_close, current_slope=hour_slope, hist_window=rev_hist_lb, slope_window=ntd_window, horizon=rev_horizon
) if len(hourly_close) else float("nan")

# Hourly indicators
roc_h = compute_roc(hourly_close, n=mom_lb_hourly) if (show_mom_hourly and len(hourly_close)) else pd.Series(index=hourly_close.index, dtype=float)
nmacd_h, nsignal_h, nhist_h = compute_nmacd(hourly_close) if len(hourly_close) else (pd.Series(dtype=float),)*3
nrsi_h = compute_nrsi(hourly_close, period=nrsi_period) if (show_nrsi and len(hourly_close)) else pd.Series(index=hourly_close.index, dtype=float)
ntd_h = compute_normalized_trend(hourly_close, window=ntd_window) if (show_ntd and len(hourly_close)) else pd.Series(index=hourly_close.index, dtype=float)
npx_h = compute_normalized_price(hourly_close, window=ntd_window) if (show_ntd and show_npx_ntd and len(hourly_close)) else pd.Series(index=hourly_close.index, dtype=float)

# Hourly BBands + HMA + signals
bb_mid_h = bb_up_h = bb_lo_h = pctb_h = nbb_h = pd.Series(index=hourly_close.index, dtype=float)
if show_bbands and len(hourly_close) > bb_win:
    bb_mid_h, bb_up_h, bb_lo_h, pctb_h, nbb_h = compute_bbands(hourly_close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

hma_h = compute_hma(hourly_close, period=hma_period) if (show_hma and len(hourly_close)) else pd.Series(index=hourly_close.index, dtype=float)

# MACD (classic) for optional chart + signal logic
macd_h, macd_sig_h, macd_hist_h = compute_macd(hourly_close) if len(hourly_close) else (pd.Series(dtype=float),)*3

macd_sr_sig = None
if len(hourly_close) and len(hma_h.dropna()) and len(macd_h.dropna()) and len(sup_h.dropna()) and len(res_h.dropna()):
    macd_sr_sig = find_macd_hma_sr_signal(
        close=hourly_close,
        hma=hma_h,
        macd=macd_h,
        sup=sup_h,
        res=res_h,
        global_trend_slope=global_slope,
        prox=sr_prox_pct
    )

# Hourly supertrend + PSAR
st_df = compute_supertrend(hourly_ohlc, atr_period=atr_period, atr_mult=atr_mult) if hourly_ohlc is not None and len(hourly_ohlc) else pd.DataFrame()
psar_df = compute_psar_from_ohlc(hourly_ohlc, step=psar_step, max_step=psar_max) if (show_psar and hourly_ohlc is not None and len(hourly_ohlc)) else pd.DataFrame()

# Hourly fib signals too (optional markers)
fib_buy_h = fib_sell_h = None
fibs_h = {}
if show_fibs and len(hourly_close) > 10 and len(npx_h.dropna()):
    fib_buy_h, fib_sell_h, fibs_h = fib_npx_zero_cross_signal_masks(
        price=hourly_close, npx=npx_h, horizon_bars=rev_horizon, proximity_pct_of_range=0.02, npx_level=0.0
    )

# Hourly slope-trigger after band reversal
slope_trigger_hourly = None
if len(hourly_close) >= max(20, slope_lb_hourly) and not hour_yhat.empty:
    slope_trigger_hourly = find_slope_trigger_after_band_reversal(
        price=hourly_close, yhat=hour_yhat, upper_band=hour_up, lower_band=hour_lo, horizon=rev_horizon
    )

# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Top summary cards
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", fmt_price_val(last_close))
c2.metric("Bull/Bear (lookback)", "Bull" if (np.isfinite(bb_chg) and bb_chg >= 0) else "Bear")
c3.metric("Global Trend (slope)", fmt_slope(global_slope))
c4.metric("Global Fit (R²)", fmt_r2(global_r2, digits=1))

st.markdown(f"**Trade Guidance (Daily):** {trade_text_daily}")

if show_fibs:
    st.caption(FIB_ALERT_TEXT)

# ---------------------------
# Plot: Daily price
# ---------------------------
def plot_daily_price():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"{ticker} — Daily Price")

    ax.plot(close_daily.index, close_daily.values, label="Close")

    # Global trendline (full history) — overlay only over current view index
    if isinstance(global_yhat, pd.Series) and not global_yhat.empty:
        gy = global_yhat.reindex(close_daily.index)
        if gy.notna().any():
            ax.plot(gy.index, gy.values, linestyle="--", linewidth=2.2, label=f"Global Trend ({fmt_slope(global_slope)})")

    # Local regression trend + band
    if isinstance(daily_yhat, pd.Series) and not daily_yhat.empty:
        ax.plot(daily_yhat.index, daily_yhat.values, linestyle="--", linewidth=2.0, label=f"Local Trend ({fmt_slope(daily_slope)} | R² {fmt_r2(daily_r2,1)})")
    if isinstance(daily_up, pd.Series) and isinstance(daily_lo, pd.Series) and (not daily_up.empty) and (not daily_lo.empty):
        ax.plot(daily_up.index, daily_up.values, linestyle=":", linewidth=1.6, label="+2σ band")
        ax.plot(daily_lo.index, daily_lo.values, linestyle=":", linewidth=1.6, label="-2σ band")

    # BBands
    if show_bbands and (not bb_mid_d.empty):
        ax.plot(bb_mid_d.index, bb_mid_d.values, linewidth=1.2, alpha=0.9, label="BB Mid")
        ax.plot(bb_up_d.index, bb_up_d.values, linewidth=1.2, alpha=0.9, label="BB Upper")
        ax.plot(bb_lo_d.index, bb_lo_d.values, linewidth=1.2, alpha=0.9, label="BB Lower")

    # HMA
    if show_hma and (not hma_d.dropna().empty):
        ax.plot(hma_d.index, hma_d.values, linewidth=2.0, label=f"HMA({hma_period})")

    # Daily S/R
    if len(sup_d.dropna()):
        ax.plot(sup_d.index, sup_d.values, linestyle="--", linewidth=1.4, alpha=0.9, label=f"Support ({sr_lb_daily})")
    if len(res_d.dropna()):
        ax.plot(res_d.index, res_d.values, linestyle="--", linewidth=1.4, alpha=0.9, label=f"Resistance ({sr_lb_daily})")

    # Fibonacci levels
    if show_fibs and isinstance(fibs_d, dict) and fibs_d:
        for k, v in fibs_d.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=0.9, alpha=0.35, linestyle="-", label=f"Fib {k}")

    # NEW: Fibonacci BUY/SELL markers (Daily price chart)
    if show_fibs and fib_buy_d is not None and fib_sell_d is not None:
        overlay_fib_npx_signals(ax, close_daily, fib_buy_d, fib_sell_d)

    # Slope trigger leaderline + legend (Daily)
    if slope_trigger_daily is not None:
        annotate_slope_trigger(ax, slope_trigger_daily)

    # Reverse Possible label (Daily)
    if show_fibs and isinstance(reg_fib_slope_rev_daily, dict):
        annotate_reverse_possible(ax, reg_fib_slope_rev_daily, text="Reverse Possible")

    style_axes(ax)
    ax.set_ylabel("Price")
    dedup_legend(ax, loc="best")
    fig.tight_layout()
    return fig

# ---------------------------
# Plot: Intraday (gapless x-axis)
# ---------------------------
def plot_intraday_gapless():
    if intraday_ohlc is None or intraday_ohlc.empty or "Close" not in intraday_ohlc.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title("Intraday (no data)")
        style_axes(ax)
        fig.tight_layout()
        return fig

    df = intraday_ohlc.copy()
    real_t = df.index
    x = np.arange(len(df), dtype=int)
    c = _coerce_1d_series(df["Close"]).astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"{ticker} — Intraday (5m, gapless)")

    ax.plot(x, c.values, label="Close (gapless)")

    # Add session markers
    if show_sessions_pst and len(real_t):
        ses = _session_lines_pst_for_day(real_t.max(), mode=mode)
        pos = _map_times_to_bar_positions(real_t, [t for t, _ in ses])
        for (ts, lbl), xp in zip(ses, pos):
            ax.axvline(x=xp, linestyle="--", linewidth=1.2, alpha=0.35)
            ax.text(xp, np.nanmax(c.values), f" {lbl}", rotation=90, va="top", fontsize=8, alpha=0.8)

    # Fibonacci levels on intraday (optional)
    if show_fibs and len(c.dropna()) > 10:
        fibs_i = fibonacci_levels(c)
        if fibs_i:
            for k, v in fibs_i.items():
                if np.isfinite(v):
                    ax.axhline(v, linewidth=0.9, alpha=0.28, linestyle="-", label=f"Fib {k}")

    # Ticks
    _apply_compact_time_ticks(ax, real_t, n_ticks=7)

    style_axes(ax)
    ax.set_ylabel("Price")
    dedup_legend(ax, loc="best")
    fig.tight_layout()
    return fig

# ---------------------------
# Plot: Hourly analysis price chart
# ---------------------------
def plot_hourly_price():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_title(f"{ticker} — Hourly (Indicators)")

    if hourly_close.empty:
        ax.text(0.5, 0.5, "Hourly data unavailable", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax)
        fig.tight_layout()
        return fig

    ax.plot(hourly_close.index, hourly_close.values, label="Close")

    # Hourly regression band
    if not hour_yhat.empty:
        ax.plot(hour_yhat.index, hour_yhat.values, linestyle="--", linewidth=2.0, label=f"Hourly Trend ({fmt_slope(hour_slope)} | R² {fmt_r2(hour_r2,1)})")
    if not hour_up.empty and not hour_lo.empty:
        ax.plot(hour_up.index, hour_up.values, linestyle=":", linewidth=1.5, label="+2σ band")
        ax.plot(hour_lo.index, hour_lo.values, linestyle=":", linewidth=1.5, label="-2σ band")

    # Hourly S/R
    if len(sup_h.dropna()):
        ax.plot(sup_h.index, sup_h.values, linestyle="--", linewidth=1.4, alpha=0.9, label=f"Support ({sr_lb_hourly})")
    if len(res_h.dropna()):
        ax.plot(res_h.index, res_h.values, linestyle="--", linewidth=1.4, alpha=0.9, label=f"Resistance ({sr_lb_hourly})")

    # BBands
    if show_bbands and (not bb_mid_h.empty):
        ax.plot(bb_mid_h.index, bb_mid_h.values, linewidth=1.1, alpha=0.9, label="BB Mid")
        ax.plot(bb_up_h.index, bb_up_h.values, linewidth=1.1, alpha=0.9, label="BB Upper")
        ax.plot(bb_lo_h.index, bb_lo_h.values, linewidth=1.1, alpha=0.9, label="BB Lower")

    # HMA
    if show_hma and (not hma_h.dropna().empty):
        ax.plot(hma_h.index, hma_h.values, linewidth=2.0, label=f"HMA({hma_period})")

    # Supertrend line
    if isinstance(st_df, pd.DataFrame) and (not st_df.empty) and "ST" in st_df.columns:
        ax.plot(st_df.index, st_df["ST"].values, linewidth=1.8, alpha=0.9, label="Supertrend")

    # PSAR
    if show_psar and isinstance(psar_df, pd.DataFrame) and (not psar_df.empty) and "PSAR" in psar_df.columns:
        ax.scatter(psar_df.index, psar_df["PSAR"].values, s=18, alpha=0.85, label="PSAR")

    # Fibonacci (hourly)
    if show_fibs and isinstance(fibs_h, dict) and fibs_h:
        for k, v in fibs_h.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=0.9, alpha=0.25, linestyle="-", label=f"Fib {k}")

    # NEW: Fibonacci BUY/SELL markers (Hourly price chart)
    if show_fibs and (fib_buy_h is not None) and (fib_sell_h is not None) and len(hourly_close):
        overlay_fib_npx_signals(ax, hourly_close, fib_buy_h, fib_sell_h)

    # Slope trigger leaderline + legend (Hourly)
    if slope_trigger_hourly is not None:
        annotate_slope_trigger(ax, slope_trigger_hourly)

    # MACD/HMA/SR star marker
    if macd_sr_sig is not None:
        annotate_macd_signal(ax, macd_sr_sig["time"], macd_sr_sig["price"], macd_sr_sig["side"])

    style_axes(ax)
    ax.set_ylabel("Price")
    dedup_legend(ax, loc="best")
    fig.tight_layout()
    return fig

# ---------------------------
# Plot: NTD panel (hourly)
# ---------------------------
def plot_hourly_ntd_panel():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_title("Hourly NTD Panel")

    if hourly_close.empty or (not show_ntd) or ntd_h.empty:
        ax.text(0.5, 0.5, "NTD unavailable", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax)
        fig.tight_layout()
        return fig

    ax.plot(ntd_h.index, ntd_h.values, linewidth=2.0, label="NTD")

    if shade_ntd:
        shade_ntd_regions(ax, ntd_h)

    # NPX overlay + crosses
    if show_npx_ntd and not npx_h.empty:
        overlay_npx_on_ntd(ax, npx_h, ntd_h, mark_crosses=mark_npx_cross)

    # Trend-based triangles + channel shading
    if show_nrsi:
        # Use hourly slope for directional triangles
        overlay_ntd_triangles_by_trend(ax, ntd_h, hour_slope)

    # Reversal stars (support/resistance + NTD)
    if show_nrsi and len(sup_h.dropna()) and len(res_h.dropna()):
        overlay_ntd_sr_reversal_stars(
            ax, price=hourly_close, sup=sup_h, res=res_h,
            trend_slope=hour_slope, ntd=ntd_h,
            prox=sr_prox_pct, bars_confirm=rev_bars_confirm
        )

    # HMA reversal markers on NTD
    if show_hma_rev_ntd and show_hma and (not hma_h.dropna().empty):
        overlay_hma_reversal_on_ntd(ax, hourly_close, hma_h, lookback=hma_rev_lb, period=hma_period, ntd=ntd_h)

    # NTD channel highlight between S/R on NTD (simple background cue)
    if show_ntd_channel and len(sup_h.dropna()) and len(res_h.dropna()) and len(ntd_h.dropna()):
        # When price is between S/R, shade lightly
        c = hourly_close.reindex(ntd_h.index)
        s = sup_h.reindex(ntd_h.index).ffill()
        r = res_h.reindex(ntd_h.index).ffill()
        ok = c.notna() & s.notna() & r.notna()
        between = (c >= s) & (c <= r)
        between = between.reindex(ntd_h.index, fill_value=False)
        ax.fill_between(ntd_h.index, -1.05, 1.05, where=between.values, alpha=0.05)

    ax.axhline(0.0, linewidth=1.2, alpha=0.35, linestyle="--")

    ax.set_ylim(-1.05, 1.05)
    style_axes(ax)
    ax.set_ylabel("NTD (tanh)")
    dedup_legend(ax, loc="best")
    fig.tight_layout()
    return fig

# ---------------------------
# Plot: Optional MACD chart (hourly)
# ---------------------------
def plot_macd_chart():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_title("Hourly MACD")

    if hourly_close.empty or macd_h.empty:
        ax.text(0.5, 0.5, "MACD unavailable", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax)
        fig.tight_layout()
        return fig

    ax.plot(macd_h.index, macd_h.values, linewidth=1.8, label="MACD")
    ax.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.5, alpha=0.9, label="Signal")
    ax.axhline(0.0, linewidth=1.2, alpha=0.35, linestyle="--")

    style_axes(ax)
    ax.set_ylabel("MACD")
    dedup_legend(ax, loc="best")
    fig.tight_layout()
    return fig

# ---------------------------
# Plot: Forecast
# ---------------------------
def plot_forecast():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_title(f"{ticker} — 30-Day SARIMAX Forecast")

    s = _coerce_1d_series(df_hist).dropna()
    if s.empty:
        ax.text(0.5, 0.5, "No historical data", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax)
        fig.tight_layout()
        return fig

    ax.plot(s.index, s.values, label="History")

    if fc_idx is not None and len(fc_idx) and fc_vals is not None and len(fc_vals):
        ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, linestyle="--", linewidth=2.0, label="Forecast")
        if isinstance(fc_ci, pd.DataFrame) and not fc_ci.empty:
            try:
                lo = fc_ci.iloc[:, 0].to_numpy(dtype=float)
                hi = fc_ci.iloc[:, 1].to_numpy(dtype=float)
                ax.fill_between(fc_idx, lo, hi, alpha=0.12, label="Conf. Int.")
            except Exception:
                pass

    style_axes(ax)
    ax.set_ylabel("Price")
    dedup_legend(ax, loc="best")
    fig.tight_layout()
    return fig

# ---------------------------
# Tabs (Ribbon styling handled by CSS above)
# ---------------------------
tabs = st.tabs(["📅 Daily", "⏱️ Intraday (5m)", "🧠 Hourly + NTD", "🔮 Forecast", "🧾 Signals"])

with tabs[0]:
    st.pyplot(plot_daily_price())

with tabs[1]:
    st.pyplot(plot_intraday_gapless())

with tabs[2]:
    st.pyplot(plot_hourly_price())
    st.pyplot(plot_hourly_ntd_panel())
    if show_mom_hourly and (not roc_h.empty):
        fig, ax = plt.subplots(figsize=(12, 3.0))
        ax.set_title("Hourly Momentum (ROC%)")
        ax.plot(roc_h.index, roc_h.values, linewidth=1.8, label=f"ROC({mom_lb_hourly})")
        ax.axhline(0.0, linewidth=1.2, alpha=0.35, linestyle="--")
        style_axes(ax)
        dedup_legend(ax, loc="best")
        fig.tight_layout()
        st.pyplot(fig)

    if show_macd:
        st.pyplot(plot_macd_chart())

with tabs[3]:
    st.pyplot(plot_forecast())

with tabs[4]:
    st.subheader("Daily Signals")
    st.write(f"- Daily slope: **{fmt_slope(daily_slope)}** | R²: **{fmt_r2(daily_r2,1)}**")
    st.write(f"- Global slope: **{fmt_slope(global_slope)}** | R²: **{fmt_r2(global_r2,1)}**")
    st.write(f"- Reversal probability (daily): **{fmt_pct(rev_prob_daily, 1)}**")
    if show_fibs and fib_rev_daily is not None:
        st.success(
            f"Fib reversal trigger: **{fib_rev_daily.get('side','')}** "
            f"(from {fib_rev_daily.get('from_level','')} at {fib_rev_daily.get('touch_time','')})"
        )

    st.subheader("Hourly Signals")
    st.write(f"- Hourly slope: **{fmt_slope(hour_slope)}** | R²: **{fmt_r2(hour_r2,1)}**")
    st.write(f"- Reversal probability (hourly): **{fmt_pct(rev_prob_hourly, 1)}**")

    if macd_sr_sig is not None:
        st.info(f"MACD/HMA/SR signal: **{macd_sr_sig['side']}** at {macd_sr_sig['time']} (px {fmt_price_val(macd_sr_sig['price'])})")

    if slope_trigger_daily is not None:
        st.info(
            f"Daily slope trigger: **{slope_trigger_daily['side']}** "
            f"touch {slope_trigger_daily['touch_time']} → cross {slope_trigger_daily['cross_time']}"
        )
    if slope_trigger_hourly is not None:
        st.info(
            f"Hourly slope trigger: **{slope_trigger_hourly['side']}** "
            f"touch {slope_trigger_hourly['touch_time']} → cross {slope_trigger_hourly['cross_time']}"
        )
