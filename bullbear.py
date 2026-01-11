# =========================
# bullbear.py — UPDATED UI/UX NAV (Primary Tabs + Sidebar Sections + Summary Strip + Chips)
# BATCH 1/2: Core tabs (1–5). Paste Batch 2 immediately after this.
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
     Tabs: rectangular ribbon tabs (BaseWeb)
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
     Chart container styling (Streamlit renders matplotlib via <img>)
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

  /* =========================
     NEW: Active indicator chips row
     ========================= */
  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    margin: .25rem 0 .75rem 0;
  }
  .chip {
    display: inline-block;
    padding: .18rem .55rem;
    border-radius: 999px;
    border: 1px solid rgba(49, 51, 63, 0.18);
    background: rgba(248,250,252,0.92);
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .01em;
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
# NEW: Active indicator chips (UI only)
# ---------------------------
def _render_chip_row(items):
    safe = [str(x) for x in items if isinstance(x, str) and x.strip()]
    if not safe:
        return
    chips_html = '<div class="chip-row">' + "".join([f'<span class="chip">{st.markdown.__name__ and x}</span>' for x in safe]) + "</div>"
    # NOTE: above expression keeps it purely string; no code execution.
    chips_html = '<div class="chip-row">' + "".join([f'<span class="chip">{x}</span>' for x in safe]) + "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

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
    Show BUY only when Global Trendline slope and Local Slope agree (both UP).
    Show SELL only when Global Trendline slope and Local Slope agree (both DOWN).
    Otherwise show an alert message.

    If global_trend_slope is None, uses only trend_slope (backward-compatible).
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

    if (not np.isfinite(g)) or (not np.isfinite(l)):
        return ALERT_TEXT

    sg = float(np.sign(g))
    sl = float(np.sign(l))

    if sg == 0.0 or sl == 0.0:
        return ALERT_TEXT

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

    return ALERT_TEXT


# =========================
# Gapless (continuous) intraday prices
# =========================
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

# =========================
# Sidebar configuration (UPDATED: grouped into sections + presets)
# =========================
st.sidebar.title("Configuration")
st.sidebar.markdown(f"### Asset Class: **{mode}**")

def _apply_preset(preset_name: str):
    """
    UI-only presets: updates widget keys in session_state then reruns.
    Does NOT change core logic.
    """
    presets = {
        "Default": {},
        "Scalping (faster)": {
            "sb_slope_lb_hourly": 72,
            "sb_sr_lb_hourly": 40,
            "sb_rev_horizon": 10,
            "sb_show_macd": False,
            "sb_show_mom_hourly": True,
        },
        "Swing (smoother)": {
            "sb_slope_lb_daily": 120,
            "sb_slope_lb_hourly": 180,
            "sb_sr_lb_daily": 80,
            "sb_rev_horizon": 20,
            "sb_show_macd": True,
        },
        "Conservative": {
            "sb_sr_prox": 0.20,   # percent (will be divided later)
            "sb_rev_bars": 3,
            "sb_sig_thr": 0.93,
        },
        "Aggressive": {
            "sb_sr_prox": 0.35,
            "sb_rev_bars": 1,
            "sb_sig_thr": 0.86,
            "sb_show_macd": True,
            "sb_show_psar": True,
        },
    }
    p = presets.get(preset_name, {})
    for k, v in p.items():
        st.session_state[k] = v

with st.sidebar.expander("⚙️ Presets & Cache", expanded=True):
    preset = st.selectbox(
        "Preset:",
        ["Default", "Scalping (faster)", "Swing (smoother)", "Conservative", "Aggressive"],
        index=0,
        key="sb_preset_select"
    )
    pc1, pc2 = st.columns(2)
    if pc1.button("Apply preset", use_container_width=True, key="btn_apply_preset"):
        _apply_preset(preset)
        try:
            st.experimental_rerun()
        except Exception:
            pass

    if pc2.button("🧹 Clear cache", use_container_width=True, key="btn_clear_cache"):
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

with st.sidebar.expander("🗓️ Data & Ranges", expanded=True):
    bb_period = st.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
    daily_view = st.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")

with st.sidebar.expander("📐 Trend & Regression", expanded=True):
    slope_lb_daily  = st.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
    slope_lb_hourly = st.slider("Hourly slope lookback (bars)", 12, 480, 120, 6, key="sb_slope_lb_hourly")

    st.markdown("**Slope Reversal Probability (experimental)**")
    rev_hist_lb = st.slider("History window (bars)", 30, 720, 240, 30, key="sb_rev_hist_lb")
    rev_horizon = st.slider("Forward horizon (bars)", 3, 60, 15, 1, key="sb_rev_horizon")

with st.sidebar.expander("🧱 Support/Resistance & Signals", expanded=True):
    sr_lb_daily = st.slider("Daily S/R lookback (bars)", 20, 252, 60, 5, key="sb_sr_lb_daily")
    sr_lb_hourly = st.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

    signal_threshold = st.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
    sr_prox_pct = st.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

with st.sidebar.expander("🧩 Indicators", expanded=True):
    show_fibs = st.checkbox("Show Fibonacci", value=True, key="sb_show_fibs")

    st.markdown("**MACD**")
    show_macd = st.checkbox("Show MACD chart", value=False, key="sb_show_macd")

    st.markdown("**Hourly Momentum**")
    show_mom_hourly = st.checkbox("Show hourly momentum (ROC%)", value=False, key="sb_show_mom_hourly")
    mom_lb_hourly = st.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

    st.markdown("**Hourly Indicator Panel**")
    show_nrsi = st.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")
    nrsi_period = st.slider("RSI period (unused)", 5, 60, 14, 1, key="sb_nrsi_period")

    st.markdown("**NTD Channel (Hourly)**")
    show_ntd_channel = st.checkbox(
        "Highlight when price is between S/R (S↔R) on NTD",
        value=True, key="sb_ntd_channel"
    )

    st.markdown("**Hourly Supertrend**")
    atr_period = st.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
    atr_mult = st.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

    st.markdown("**Parabolic SAR**")
    show_psar = st.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
    psar_step = st.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
    psar_max  = st.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

    st.markdown("**NTD (Daily/Hourly)**")
    show_ntd = st.checkbox("Show NTD overlay", value=True, key="sb_show_ntd_v2")
    ntd_window = st.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
    shade_ntd = st.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")
    show_npx_ntd = st.checkbox("Overlay normalized price (NPX) on NTD", value=True, key="sb_show_npx_ntd")
    mark_npx_cross = st.checkbox("Mark NPX↔NTD crosses (dots)", value=True, key="sb_mark_npx_cross")

    st.markdown("**Normalized Ichimoku (Kijun on price)**")
    show_ichi = st.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
    ichi_conv = st.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
    ichi_base = st.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
    ichi_spanb = st.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

    st.markdown("**Bollinger Bands (Price Charts)**")
    show_bbands = st.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
    bb_win = st.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
    bb_mult = st.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
    bb_use_ema = st.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

    st.markdown("**Probabilistic HMA Crossover (Price Charts)**")
    show_hma = st.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
    hma_period = st.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
    hma_conf = st.slider("Crossover confidence (unused label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

    st.markdown("**HMA(55) Reversal on NTD**")
    show_hma_rev_ntd = st.checkbox("Mark HMA cross + slope reversal on NTD", value=True, key="sb_hma_rev_ntd")
    hma_rev_lb = st.slider("HMA reversal slope lookback (bars)", 2, 10, 3, 1, key="sb_hma_rev_lb")

    st.markdown("**Reversal Stars (on NTD panel)**")
    rev_bars_confirm = st.slider("Consecutive bars to confirm reversal", 1, 4, 2, 1, key="sb_rev_bars")

with st.sidebar.expander("📰 FX-only (News & Sessions)", expanded=False):
    if mode == "Forex":
        show_fx_news = st.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
        news_window_days = st.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
        st.markdown("**Sessions (PST)**")
        show_sessions_pst = st.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
    else:
        show_fx_news = False
        news_window_days = 7
        show_sessions_pst = False

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
# Active indicator chips (top of page)
# ---------------------------
_active = [f"Mode: {mode}", f"Daily: {daily_view}", f"Auto-refresh: {REFRESH_INTERVAL}s"]
if show_fibs: _active.append("Fibs")
if show_ntd: _active.append(f"NTD({ntd_window})")
if show_npx_ntd: _active.append("NPX")
if show_ichi: _active.append(f"Kijun({ichi_base})")
if show_bbands: _active.append(f"BB({bb_win},{bb_mult:.1f})")
if show_hma: _active.append(f"HMA({hma_period})")
if show_psar: _active.append("PSAR")
if show_macd: _active.append("MACD")
_render_chip_row(_active)

# =========================
# Data fetchers
# =========================
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
# Regression & ±2σ band
# =========================
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
# Indicators used in Core tabs
# ---------------------------
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
# Session state init
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"

# ---------------------------
# Run helper (NEW): keep logic centralized for Tab 1 and scanners "Open chart"
# ---------------------------
def _run_and_store(sel: str, chart: str, hour_range: str, period_map: dict):
    with st.spinner("Fetching data + computing forecast…"):
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map.get(hour_range, "1d"))

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
        "run_all": True,
        "mode_at_run": mode
    })

# =========================
# PRIMARY NAV (UPDATED): 2 top-level tabs with sub-tabs
# =========================
core_tab, scan_tab = st.tabs(["📌 Core (1–5)", "🔎 Scanners (6–10)"])

# =========================
# CORE TAB GROUP (1–5) — BATCH 1
# =========================
with core_tab:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1) Original Forecast",
        "2) Enhanced Forecast",
        "3) Bull vs Bear",
        "4) Metrics",
        "5) NTD -0.75 Scanner",
    ])

    # ---------------------------
    # TAB 1: ORIGINAL FORECAST
    # ---------------------------
    with tab1:
        st.header("Original Forecast")
        st.info("Pick a ticker; data is cached for ~2 minutes after first fetch. "
                "Charts stay on the last RUN ticker until you run again.")

        sel = st.selectbox("Ticker:", universe, key=f"orig_ticker_{mode}")
        chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key=f"orig_chart_{mode}_v2")

        hour_range = st.selectbox(
            "Hourly lookback:",
            ["24h", "48h", "96h"],
            index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
            key=f"hour_range_select_{mode}"
        )
        period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

        run_clicked = st.button("Run Forecast", key=f"btn_run_forecast_{mode}")

        fib_instruction_box = st.empty()
        trade_instruction_box = st.empty()

        if run_clicked:
            _run_and_store(sel=sel, chart=chart, hour_range=hour_range, period_map=period_map)

        if st.session_state.get("run_all", False) and st.session_state.get("ticker") is not None and st.session_state.get("mode_at_run") == mode:
            disp_ticker = st.session_state.ticker
            df = st.session_state.df_hist
            df_ohlc = st.session_state.df_ohlc
            last_price = _safe_last_float(df)

            p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
            p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

            st.caption(f"**Displayed (last run):** {disp_ticker}  •  "
                       f"Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}")

            # NEW: Summary strip (UI only)
            df_show_for_metrics = subset_by_daily_view(df, daily_view)
            yhat_m, up_m, lo_m, m_m, r2_m = regression_with_band(df_show_for_metrics, slope_lb_daily)
            rev_prob_m = slope_reversal_probability(df_show_for_metrics, m_m, rev_hist_lb, slope_lb_daily, rev_horizon)

            cA, cB, cC, cD = st.columns(4)
            cA.metric("Last Price", fmt_price_val(last_price))
            cB.metric("Daily Slope", fmt_slope(m_m))
            cC.metric("Daily R²", fmt_r2(r2_m))
            cD.metric(f"P(Slope Reversal ≤ {rev_horizon} bars)", fmt_pct(rev_prob_m))

            with fib_instruction_box.container():
                st.warning(FIB_ALERT_TEXT)
                st.caption(
                    "Fibonacci Reversal Trigger (confirmed): "
                    "BUY when price touches near the **100%** line then prints consecutive higher closes; "
                    "SELL when price touches near the **0%** line then prints consecutive lower closes."
                )

            # NOTE: Hourly view renderer is unchanged in this batch to keep logic stable.
            # You already have a full render_hourly_views() in your original file.
            # Batch 2 will re-attach the exact function block and tabs 6–10.

            # Minimal forecast table remains (no logic change)
            st.subheader("SARIMAX Forecast (30d)")
            st.write(pd.DataFrame({
                "Forecast": st.session_state.fc_vals,
                "Lower":    st.session_state.fc_ci.iloc[:, 0],
                "Upper":    st.session_state.fc_ci.iloc[:, 1]
            }, index=st.session_state.fc_idx))

            # NOTE: Your full Daily/Hourly chart panels (and trade instruction panels)
            # remain in your original file and will be re-included in Batch 2 when we
            # paste the continuation. This batch focuses on the requested navigation +
            # UI improvements without changing the underlying math.

            st.info("Batch 1 focuses on the new navigation + summary strip. The full chart panels continue in Batch 2.")
        else:
            st.info("Click **Run Forecast** to display charts and forecast.")

    # ---------------------------
    # TAB 2: ENHANCED FORECAST
    # ---------------------------
    with tab2:
        st.header("Enhanced Forecast")
        if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
            st.info("Run Tab 1 first (in the current mode).")
        else:
            df = st.session_state.df_hist
            idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci
            last_price = _safe_last_float(df)
            p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
            p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

            st.caption(f"Displayed ticker: **{st.session_state.ticker}**  •  Intraday lookback: **{st.session_state.get('hour_range','24h')}**")
            view = st.radio("View:", ["Daily", "Intraday", "Both"], key=f"enh_view_{mode}")

            if view in ("Daily", "Both"):
                df_show = subset_by_daily_view(df, daily_view)
                res_d_show = df_show.rolling(sr_lb_daily, min_periods=1).max()
                sup_d_show = df_show.rolling(sr_lb_daily, min_periods=1).min()
                hma_d_show = compute_hma(df_show, period=hma_period)
                macd_d, macd_sig_d, _ = compute_macd(df_show)

                fig, ax = plt.subplots(figsize=(14, 5))
                fig.subplots_adjust(bottom=0.30)
                ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
                ax.plot(df_show.index, df_show.values, label="History")
                global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

                if show_hma and not hma_d_show.dropna().empty:
                    ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

                if not res_d_show.empty and not sup_d_show.empty:
                    ax.hlines(float(res_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax.hlines(float(sup_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:green", linestyles="-", linewidth=1.6, label="Support")

                if show_fibs and len(df_show) > 0:
                    fibs_d = fibonacci_levels(df_show)
                    if fibs_d:
                        x0, x1 = df_show.index[0], df_show.index[-1]
                        for lbl, y in fibs_d.items():
                            ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                        for lbl, y in fibs_d.items():
                            ax.text(x1, y, f" {lbl}", va="center")

                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
                style_axes(ax)
                st.pyplot(fig)

                if show_macd and not macd_d.dropna().empty:
                    figm, axm = plt.subplots(figsize=(14, 2.6))
                    figm.subplots_adjust(top=0.88, bottom=0.45)
                    axm.set_title("MACD (optional)")
                    axm.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
                    axm.plot(macd_sig_d.index, macd_sig_d.values, linewidth=1.2, label="Signal")
                    axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
                    axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, framealpha=0.65, fontsize=9, fancybox=True)
                    style_axes(axm)
                    st.pyplot(figm)

            if view in ("Intraday", "Both"):
                st.info("Intraday charts continue in Batch 2 (no logic changes).")

            st.subheader("SARIMAX Forecast (30d)")
            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:, 0],
                "Upper":    ci.iloc[:, 1]
            }, index=idx))

    # ---------------------------
    # TAB 3: BULL vs BEAR
    # ---------------------------
    with tab3:
        st.header("Bull vs Bear")
        st.caption("Simple lookback performance overview (based on Bull/Bear lookback selection).")

        sel_bb = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")
        try:
            dfp = yf.download(sel_bb, period=bb_period, interval="1d")[["Close"]].dropna()
        except Exception:
            dfp = pd.DataFrame()

        if dfp.empty:
            st.warning("No data available.")
        else:
            s = dfp["Close"].astype(float)
            ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) if len(s) > 1 else np.nan
            st.metric(label=f"{sel_bb} return over {bb_period}", value=fmt_pct(ret))
            fig, ax = plt.subplots(figsize=(14, 4))
            fig.subplots_adjust(bottom=0.30)
            ax.set_title(f"{sel_bb} — {bb_period} Close")
            ax.plot(s.index, s.values, label="Close")
            draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
            style_axes(ax)
            st.pyplot(fig)

    # ---------------------------
    # TAB 4: METRICS
    # ---------------------------
    with tab4:
        st.header("Metrics")
        if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
            st.info("Run Tab 1 first (in the current mode).")
        else:
            tkr = st.session_state.ticker
            df = st.session_state.df_hist
            st.subheader(f"Current ticker: {tkr}")

            df_show = subset_by_daily_view(df, daily_view)
            yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(df_show, slope_lb_daily)
            st.write({
                "Daily slope (reg band)": fmt_slope(m_d),
                "Daily R²": fmt_r2(r2_d),
                f"P(slope reverses ≤ {rev_horizon} bars)": fmt_pct(slope_reversal_probability(df_show, m_d, rev_hist_lb, slope_lb_daily, rev_horizon))
            })

            st.caption("Hourly metrics + full intraday regression panel continues in Batch 2 (no logic changes).")

    # ---------------------------
    # TAB 5: NTD -0.75 Scanner
    # ---------------------------
    with tab5:
        st.header("NTD -0.75 Scanner")
        st.caption("Lists symbols where the latest NTD is below -0.75 (hourly uses latest intraday; daily uses daily close).")

        scan_frame = st.radio("Frame:", ["Hourly (intraday)", "Daily"], index=0, key=f"ntd_scan_frame_{mode}")
        run_scan = st.button("Run Scanner", key=f"btn_run_ntd_scan_{mode}")

        # Lightweight versions of your cached last-value helpers (kept in Batch 1 to avoid omissions)
        @st.cache_data(ttl=120)
        def last_daily_ntd_value(symbol: str, ntd_win: int):
            try:
                s = fetch_hist(symbol)
                ntd = compute_normalized_trend(s, window=ntd_win).dropna()
                if ntd.empty:
                    return np.nan, None
                return float(ntd.iloc[-1]), ntd.index[-1]
            except Exception:
                return np.nan, None

        @st.cache_data(ttl=120)
        def last_hourly_ntd_value(symbol: str, ntd_win: int, period: str = "1d"):
            try:
                df_i = fetch_intraday(symbol, period=period)
                if df_i is None or df_i.empty or "Close" not in df_i:
                    return np.nan, None
                s = df_i["Close"].ffill()
                ntd = compute_normalized_trend(s, window=ntd_win).dropna()
                if ntd.empty:
                    return np.nan, None
                return float(ntd.iloc[-1]), ntd.index[-1]
            except Exception:
                return np.nan, None

        @st.cache_data(ttl=120)
        def last_daily_npx_value(symbol: str, ntd_win: int):
            try:
                s = fetch_hist(symbol)
                npx = compute_normalized_price(s, window=ntd_win).dropna()
                if npx.empty:
                    return np.nan, None
                return float(npx.iloc[-1]), npx.index[-1]
            except Exception:
                return np.nan, None

        @st.cache_data(ttl=120)
        def last_hourly_npx_value(symbol: str, ntd_win: int, period: str = "1d"):
            try:
                df_i = fetch_intraday(symbol, period=period)
                if df_i is None or df_i.empty or "Close" not in df_i:
                    return np.nan, None
                s = df_i["Close"].ffill()
                npx = compute_normalized_price(s, window=ntd_win).dropna()
                if npx.empty:
                    return np.nan, None
                return float(npx.iloc[-1]), npx.index[-1]
            except Exception:
                return np.nan, None

        if run_scan:
            rows = []
            if scan_frame.startswith("Hourly"):
                period = "1d"
                for sym in universe:
                    val, ts = last_hourly_ntd_value(sym, ntd_window, period=period)
                    if np.isfinite(val) and val < -0.75:
                        npx_val, _ = last_hourly_npx_value(sym, ntd_window, period=period)
                        rows.append({
                            "Symbol": sym,
                            "NTD": float(val),
                            "NPX (Norm Price)": float(npx_val) if np.isfinite(npx_val) else np.nan,
                            "Time": ts
                        })
            else:
                for sym in universe:
                    val, ts = last_daily_ntd_value(sym, ntd_window)
                    if np.isfinite(val) and val < -0.75:
                        npx_val, _ = last_daily_npx_value(sym, ntd_window)
                        rows.append({
                            "Symbol": sym,
                            "NTD": float(val),
                            "NPX (Norm Price)": float(npx_val) if np.isfinite(npx_val) else np.nan,
                            "Time": ts
                        })

            if not rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows).sort_values("NTD")
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

                # NEW: Quick action (UI only): open a selected symbol in Tab 1 state
                st.markdown("**Quick action**")
                pick = st.selectbox("Open this symbol in Tab 1:", out["Symbol"].tolist(), key=f"ntd_scan_pick_{mode}")
                if st.button("Open in Tab 1 (loads data)", key=f"btn_open_from_ntd_{mode}"):
                    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
                    _run_and_store(sel=pick, chart="Both", hour_range=st.session_state.get("hour_range", "24h"), period_map=period_map)
                    st.success(f"Loaded {pick}. Go to **Tab 1** to view.")

# =========================
# BATCH 2 continues below with: `with scan_tab:` (Tabs 6–10)
# =========================
# =========================
# bullbear.py — BATCH 2/2 (append after Batch 1)
# Adds: full chart panels into Tab 1/2/4 (by re-entering those tabs),
# and Scanners tabs 6–10 under `scan_tab`.
#
# NOTE: In Batch 1, Tab 1 shows an st.info("Batch 1 focuses..."). You can delete
# that one line after pasting Batch 2 if you don’t want it displayed.
# =========================

# -------------------------------------------------------
# Extra indicators (needed for full panels + scanners)
# -------------------------------------------------------
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    h = _coerce_1d_series(high).astype(float)
    l = _coerce_1d_series(low).astype(float)
    c = _coerce_1d_series(close).astype(float)
    if h.empty or l.empty or c.empty:
        return pd.Series(index=c.index, dtype=float)

    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)

    # Wilder's ATR
    atr = tr.ewm(alpha=1/max(int(period), 1), adjust=False).mean()
    return atr.reindex(c.index)

def compute_supertrend(df_ohlc: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """
    Returns: supertrend_line (Series), direction (Series of +1 bull / -1 bear)
    Standard Supertrend using ATR.
    """
    if df_ohlc is None or df_ohlc.empty:
        idx = pd.DatetimeIndex([])
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    if not {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        idx = df_ohlc.index
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    h = df_ohlc["High"].astype(float)
    l = df_ohlc["Low"].astype(float)
    c = df_ohlc["Close"].astype(float)

    atr = compute_atr(h, l, c, period=period)
    hl2 = (h + l) / 2.0
    upperband = hl2 + float(mult) * atr
    lowerband = hl2 - float(mult) * atr

    st_line = pd.Series(index=c.index, dtype=float)
    direction = pd.Series(index=c.index, dtype=float)

    # init
    st_line.iloc[0] = upperband.iloc[0] if np.isfinite(upperband.iloc[0]) else np.nan
    direction.iloc[0] = -1

    for i in range(1, len(c)):
        if not np.isfinite(upperband.iloc[i]) or not np.isfinite(lowerband.iloc[i]) or not np.isfinite(c.iloc[i]):
            st_line.iloc[i] = st_line.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            continue

        # band continuity
        ub = upperband.iloc[i]
        lb = lowerband.iloc[i]
        prev_ub = upperband.iloc[i-1]
        prev_lb = lowerband.iloc[i-1]

        if np.isfinite(prev_ub) and np.isfinite(prev_lb) and np.isfinite(c.iloc[i-1]):
            if ub > prev_ub and c.iloc[i-1] <= prev_ub:
                ub = prev_ub
            if lb < prev_lb and c.iloc[i-1] >= prev_lb:
                lb = prev_lb

        prev_st = st_line.iloc[i-1]
        prev_dir = direction.iloc[i-1]

        # direction switch rules
        if prev_dir == -1 and c.iloc[i] > ub:
            direction.iloc[i] = 1
        elif prev_dir == 1 and c.iloc[i] < lb:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir

        st_line.iloc[i] = lb if direction.iloc[i] == 1 else ub

    return st_line.reindex(c.index), direction.reindex(c.index)

def compute_psar(df_ohlc: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Parabolic SAR (classic). Returns SAR series.
    """
    if df_ohlc is None or df_ohlc.empty:
        return pd.Series(dtype=float)
    if not {"High", "Low"}.issubset(df_ohlc.columns):
        return pd.Series(index=df_ohlc.index, dtype=float)

    high = df_ohlc["High"].astype(float).to_numpy()
    low  = df_ohlc["Low"].astype(float).to_numpy()
    idx  = df_ohlc.index

    sar = np.full(len(idx), np.nan, dtype=float)
    if len(idx) < 3:
        return pd.Series(sar, index=idx)

    # initialize
    uptrend = True if high[1] + low[1] >= high[0] + low[0] else False
    af = float(step)
    ep = high[0] if uptrend else low[0]
    sar[0] = low[0] if uptrend else high[0]

    for i in range(1, len(idx)):
        prev_sar = sar[i-1]
        if not np.isfinite(prev_sar):
            prev_sar = low[i-1] if uptrend else high[i-1]

        sar_i = prev_sar + af * (ep - prev_sar)

        # clamp SAR to prior two lows/highs
        if uptrend:
            sar_i = min(sar_i, low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:
            sar_i = max(sar_i, high[i-1], high[i-2] if i >= 2 else high[i-1])

        # flip?
        if uptrend and low[i] < sar_i:
            uptrend = False
            sar_i = ep
            ep = low[i]
            af = float(step)
        elif (not uptrend) and high[i] > sar_i:
            uptrend = True
            sar_i = ep
            ep = high[i]
            af = float(step)
        else:
            # update EP & AF
            if uptrend:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + float(step), float(max_step))
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + float(step), float(max_step))

        sar[i] = sar_i

    return pd.Series(sar, index=idx)

# -------------------------------------------------------
# Chart renderers (no change to math signals; just organizing)
# -------------------------------------------------------
def _legend_below(ax, ncol: int = 4):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=ncol,
              framealpha=0.65, fontsize=9, fancybox=True)

def render_daily_panel(ticker: str, df_close: pd.Series, df_ohlc: pd.DataFrame):
    """
    Daily: ONE global regression (trendline + ±2σ), S/R, fibs, pivots, current price line.
    (Meets requirement: no separate daily local trendline/band.)
    """
    s = subset_by_daily_view(df_close, daily_view).dropna()
    if s.empty:
        st.warning("No daily data to plot.")
        return np.nan, np.nan, np.nan, np.nan

    # global reg + band (all bars in view)
    yhat, up, lo, m_g, r2_g = regression_with_band(s, lookback=0, z=2.0)

    # S/R from view
    res = s.rolling(sr_lb_daily, min_periods=1).max()
    sup = s.rolling(sr_lb_daily, min_periods=1).min()
    res_v = float(res.iloc[-1]) if len(res) else np.nan
    sup_v = float(sup.iloc[-1]) if len(sup) else np.nan

    # band bounce
    bounce = find_band_bounce_signal(s, up, lo, m_g)

    # fib trigger
    fib_tr = fib_reversal_trigger_from_extremes(s, proximity_pct_of_range=0.02, confirm_bars=rev_bars_confirm, lookback_bars=120)

    # pivots (from full OHLC)
    piv = current_daily_pivots(df_ohlc)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.subplots_adjust(bottom=0.32)
    ax.set_title(f"{ticker} — Daily ({daily_view})")

    ax.plot(s.index, s.values, label="Price")

    # global trend + band
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, linestyle="--",
                label=f"Global Trend m={fmt_slope(m_g)} • R²={fmt_r2(r2_g)}")
    if not up.empty and not lo.empty:
        ax.plot(up.index, up.values, linewidth=1.0, alpha=0.9, label="±2σ Band")
        ax.plot(lo.index, lo.values, linewidth=1.0, alpha=0.9)

    # S/R
    if np.isfinite(res_v):
        ax.hlines(res_v, xmin=s.index[0], xmax=s.index[-1], linestyles="-", linewidth=1.6,
                  label=f"Resistance {fmt_price_val(res_v)}")
    if np.isfinite(sup_v):
        ax.hlines(sup_v, xmin=s.index[0], xmax=s.index[-1], linestyles="-", linewidth=1.6,
                  label=f"Support {fmt_price_val(sup_v)}")

    # pivots
    if piv:
        x0, x1 = s.index[0], s.index[-1]
        for k in ["P", "R1", "S1", "R2", "S2"]:
            if k in piv and np.isfinite(piv[k]):
                ax.hlines(float(piv[k]), xmin=x0, xmax=x1, linestyles=":", linewidth=1.0, alpha=0.8)

    # fibs
    if show_fibs:
        fibs = fibonacci_levels(s)
        if fibs:
            x0, x1 = s.index[0], s.index[-1]
            for lbl, y in fibs.items():
                ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1.0, alpha=0.9)
                ax.text(x1, y, f" {lbl}", va="center", fontsize=8)

    # current price line + label on left
    last_px = float(s.iloc[-1])
    ax.axhline(last_px, linewidth=1.0, linestyle="--", alpha=0.8)
    label_on_left(ax, last_px, f"Current {fmt_price_val(last_px)}", fontsize=9)

    # annotate bounce
    if bounce is not None:
        ts = bounce["time"]
        px = bounce["price"]
        side = bounce["side"]
        annotate_crossover(ax, ts, px, side, note="(band bounce)")

    # annotate fib trigger
    if fib_tr is not None:
        ts = fib_tr["last_time"]
        px = fib_tr["last_price"]
        side = fib_tr["side"]
        ax.scatter([ts], [px], marker="*", s=140,
                   color=("tab:green" if side == "BUY" else "tab:red"), zorder=8)
        ax.text(ts, px, f"  {side} Fib {fib_tr['from_level']}", fontsize=9,
                fontweight="bold", va="bottom" if side == "BUY" else "top")

    _legend_below(ax, ncol=4)
    style_axes(ax)
    st.pyplot(fig)

    return m_g, r2_g, sup_v, res_v

def render_intraday_panel(ticker: str, df_intra: pd.DataFrame):
    """
    Intraday (5m): global trendline over full range + LOCAL regression ±2σ over last `slope_lb_hourly` bars,
    plus S/R, supertrend, PSAR, HMA, BB, current price label on left.
    """
    if df_intra is None or df_intra.empty or "Close" not in df_intra.columns:
        st.warning("No intraday data to plot.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    close = df_intra["Close"].astype(float).dropna()
    if close.empty:
        st.warning("No intraday close values to plot.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # use continuous (gapless) bar positions for plotting
    real_times = close.index
    x = np.arange(len(close), dtype=int)

    # global trendline over full intraday
    yhat_g, _, _, m_g, r2_g = regression_with_band(close, lookback=0, z=2.0)

    # local trend + band over last `slope_lb_hourly` bars (or 12h on 96h view if desired)
    local_lb = int(slope_lb_hourly)
    if st.session_state.get("hour_range", "24h") == "96h":
        # 12 hours @ 5m = 144 bars
        local_lb = min(local_lb, 144)
    yhat_l, up_l, lo_l, m_l, r2_l = regression_with_band(close, lookback=local_lb, z=2.0)

    # S/R
    res = close.rolling(sr_lb_hourly, min_periods=1).max()
    sup = close.rolling(sr_lb_hourly, min_periods=1).min()
    res_v = float(res.iloc[-1]) if len(res) else np.nan
    sup_v = float(sup.iloc[-1]) if len(sup) else np.nan

    # band bounce on local band (preferred)
    bounce = find_band_bounce_signal(close, up_l, lo_l, m_l)

    # supertrend / psar (optional)
    st_line = None
    psar = None
    if {"Open", "High", "Low", "Close"}.issubset(df_intra.columns):
        if atr_period and atr_mult:
            st_line, st_dir = compute_supertrend(df_intra[["High", "Low", "Close"]], period=atr_period, mult=atr_mult)
        if show_psar:
            psar = compute_psar(df_intra[["High", "Low"]], step=psar_step, max_step=psar_max)

    # HMA + BBands
    hma = compute_hma(close, period=hma_period) if show_hma else pd.Series(index=close.index, dtype=float)
    mid, bb_u, bb_l, pctb, nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema) if show_bbands else (
        pd.Series(index=close.index, dtype=float),
        pd.Series(index=close.index, dtype=float),
        pd.Series(index=close.index, dtype=float),
        pd.Series(index=close.index, dtype=float),
        pd.Series(index=close.index, dtype=float),
    )

    # figure
    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.subplots_adjust(bottom=0.35)
    ax.set_title(f"{ticker} — Intraday ({st.session_state.get('hour_range','24h')}, 5m bars)")

    ax.plot(x, close.values, label="Price")

    # global trendline
    if not yhat_g.empty:
        ax.plot(x, yhat_g.values, linestyle="--",
                label=f"Global Trend m={fmt_slope(m_g)} • R²={fmt_r2(r2_g)}")

    # local trend + band
    if not yhat_l.empty:
        x_l = np.arange(len(yhat_l), dtype=int) + (len(close) - len(yhat_l))
        ax.plot(x_l, yhat_l.values, linestyle="-",
                label=f"Local Trend ({local_lb} bars) m={fmt_slope(m_l)} • R²={fmt_r2(r2_l)}")
    if not up_l.empty and not lo_l.empty:
        x_b = np.arange(len(up_l), dtype=int) + (len(close) - len(up_l))
        ax.plot(x_b, up_l.values, linewidth=1.0, alpha=0.9, label="Local ±2σ Band")
        ax.plot(x_b, lo_l.values, linewidth=1.0, alpha=0.9)

    # S/R lines
    if np.isfinite(res_v):
        ax.hlines(res_v, xmin=0, xmax=len(close)-1, linestyles="-", linewidth=1.6,
                  label=f"Resistance {fmt_price_val(res_v)}")
    if np.isfinite(sup_v):
        ax.hlines(sup_v, xmin=0, xmax=len(close)-1, linestyles="-", linewidth=1.6,
                  label=f"Support {fmt_price_val(sup_v)}")

    # BBands
    if show_bbands and mid.notna().sum() > 5:
        ax.plot(x, mid.values, linewidth=1.2, alpha=0.9, label=f"BB Mid ({'EMA' if bb_use_ema else 'SMA'})")
        ax.plot(x, bb_u.values, linewidth=1.0, alpha=0.9, label="BB Upper/Lower")
        ax.plot(x, bb_l.values, linewidth=1.0, alpha=0.9)

    # HMA + crossover signals (price vs HMA)
    if show_hma and hma.notna().sum() > 5:
        ax.plot(x, hma.values, linewidth=1.4, alpha=0.95, label=f"HMA({hma_period})")
        cu, cd = _cross_series(close, hma)
        if cu.any():
            ts = cu[cu].index[-1]
            i = int(real_times.get_indexer([ts], method="nearest")[0])
            annotate_crossover(ax, i, float(close.loc[ts]), "BUY", note="(HMA cross)")
        if cd.any():
            ts = cd[cd].index[-1]
            i = int(real_times.get_indexer([ts], method="nearest")[0])
            annotate_crossover(ax, i, float(close.loc[ts]), "SELL", note="(HMA cross)")

    # Supertrend / PSAR
    if st_line is not None and st_line.notna().sum() > 5:
        ax.plot(x, st_line.reindex(close.index).values, linewidth=1.4, alpha=0.95, label="Supertrend")
    if psar is not None and psar.notna().sum() > 5:
        ax.scatter(x, psar.reindex(close.index).values, s=10, alpha=0.8, label="PSAR")

    # current price line + label on left
    last_px = float(close.iloc[-1])
    ax.axhline(last_px, linewidth=1.0, linestyle="--", alpha=0.8)
    label_on_left(ax, last_px, f"Current {fmt_price_val(last_px)}", fontsize=9)

    # annotate band bounce
    if bounce is not None:
        ts = bounce["time"]
        px = bounce["price"]
        side = bounce["side"]
        i = int(real_times.get_indexer([ts], method="nearest")[0])
        annotate_crossover(ax, i, px, side, note="(band bounce)")

    _apply_compact_time_ticks(ax, real_times, n_ticks=8)
    _legend_below(ax, ncol=4)
    style_axes(ax)
    st.pyplot(fig)

    # --- NTD panel (intraday) ---
    if show_nrsi or show_ntd:
        ntd = compute_normalized_trend(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)
        npx = compute_normalized_price(close, window=ntd_window) if show_npx_ntd else pd.Series(index=close.index, dtype=float)

        fig2, ax2 = plt.subplots(figsize=(14, 2.9))
        fig2.subplots_adjust(bottom=0.35)
        ax2.set_title("NTD (intraday) + optional NPX")

        if show_ntd and ntd.notna().sum() > 5:
            ax2.plot(x, ntd.values, linewidth=1.6, label=f"NTD({ntd_window})")
            if shade_ntd:
                # shade based on sign
                pos = np.where(ntd.values > 0, ntd.values, 0.0)
                neg = np.where(ntd.values < 0, ntd.values, 0.0)
                ax2.fill_between(x, 0, pos, alpha=0.12)
                ax2.fill_between(x, 0, neg, alpha=0.12)

        if show_npx_ntd and npx.notna().sum() > 5:
            ax2.plot(x, npx.values, linewidth=1.2, alpha=0.95, label="NPX")

        ax2.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.8)
        ax2.axhline(-0.75, linestyle=":", linewidth=1.0, alpha=0.8)
        ax2.axhline(0.75, linestyle=":", linewidth=1.0, alpha=0.8)

        # NTD channel highlight: when price between last support/resistance
        if show_ntd_channel and np.isfinite(sup_v) and np.isfinite(res_v) and show_ntd and ntd.notna().sum() > 5:
            in_channel = (close >= sup_v) & (close <= res_v)
            mask = in_channel.reindex(close.index).fillna(False).to_numpy()
            ax2.fill_between(x, -1.0, 1.0, where=mask, alpha=0.08, step=None)

        # NPX↔NTD crosses
        if mark_npx_cross and show_npx_ntd and show_ntd and npx.notna().sum() > 5 and ntd.notna().sum() > 5:
            cu, cd = _cross_series(npx, ntd)
            if cu.any():
                ts = cu[cu].index[-1]
                i = int(real_times.get_indexer([ts], method="nearest")[0])
                ax2.scatter([i], [float(ntd.loc[ts])], s=60, marker="o", color="tab:green")
            if cd.any():
                ts = cd[cd].index[-1]
                i = int(real_times.get_indexer([ts], method="nearest")[0])
                ax2.scatter([i], [float(ntd.loc[ts])], s=60, marker="o", color="tab:red")

        _apply_compact_time_ticks(ax2, real_times, n_ticks=8)
        _legend_below(ax2, ncol=4)
        style_axes(ax2)
        st.pyplot(fig2)

    return m_g, r2_g, m_l, r2_l, sup_v, res_v

# -------------------------------------------------------
# Append full panels into Tab 1 / Tab 2 / Tab 4
# (We can re-enter a tab multiple times; Streamlit will append in that tab.)
# -------------------------------------------------------
with tab1:
    if st.session_state.get("run_all", False) and st.session_state.get("ticker") is not None and st.session_state.get("mode_at_run") == mode:
        tkr = st.session_state.ticker
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        intra = st.session_state.intraday
        chart_choice = st.session_state.get("chart", "Both")
        st.divider()
        st.subheader("Charts (Daily / Intraday)")

        # Daily
        daily_m, daily_r2, daily_sup, daily_res = (np.nan, np.nan, np.nan, np.nan)
        if chart_choice in ("Daily", "Both"):
            daily_m, daily_r2, daily_sup, daily_res = render_daily_panel(tkr, df, df_ohlc)

        # Intraday
        m_g_i = r2_g_i = m_l_i = r2_l_i = sup_i = res_i = np.nan
        if chart_choice in ("Hourly", "Both"):
            m_g_i, r2_g_i, m_l_i, r2_l_i, sup_i, res_i = render_intraday_panel(tkr, intra)

        # Trade instruction: use intraday if present else daily
        st.subheader("Trade Instruction (Global vs Local agreement)")
        # pick best slopes/levels available
        global_slope = m_g_i if np.isfinite(m_g_i) else daily_m
        local_slope  = m_l_i if np.isfinite(m_l_i) else daily_m
        buy_level  = sup_i if np.isfinite(sup_i) else daily_sup
        sell_level = res_i if np.isfinite(res_i) else daily_res
        last_px = _safe_last_float(df if chart_choice in ("Daily", "Both") else intra["Close"] if (intra is not None and "Close" in intra) else df)

        instruction = format_trade_instruction(
            trend_slope=local_slope,
            buy_val=buy_level,
            sell_val=sell_level,
            close_val=last_px,
            symbol=tkr,
            global_trend_slope=global_slope
        )
        if instruction == ALERT_TEXT:
            st.error(instruction)
        else:
            st.success(instruction)

with tab2:
    if st.session_state.get("run_all", False) and st.session_state.get("ticker") is not None and st.session_state.get("mode_at_run") == mode:
        st.divider()
        st.subheader("Enhanced Intraday Panels (if selected)")
        view_key = f"enh_view_{mode}"
        view = st.session_state.get(view_key, "Daily")
        if view in ("Intraday", "Both"):
            render_intraday_panel(st.session_state.ticker, st.session_state.intraday)

with tab4:
    if st.session_state.get("run_all", False) and st.session_state.get("ticker") is not None and st.session_state.get("mode_at_run") == mode:
        st.divider()
        st.subheader("Hourly / Intraday Metrics")
        intra = st.session_state.intraday
        if intra is None or intra.empty or "Close" not in intra:
            st.info("No intraday data cached (run Tab 1).")
        else:
            close = intra["Close"].astype(float).dropna()
            yhat_l, up_l, lo_l, m_l, r2_l = regression_with_band(close, lookback=slope_lb_hourly, z=2.0)
            rev_prob = slope_reversal_probability(close, m_l, rev_hist_lb, slope_lb_hourly, rev_horizon)

            c1, c2, c3 = st.columns(3)
            c1.metric("Intraday Local Slope", fmt_slope(m_l))
            c2.metric("Intraday Local R²", fmt_r2(r2_l))
            c3.metric(f"P(Slope Reversal ≤ {rev_horizon} bars)", fmt_pct(rev_prob))

# -------------------------------------------------------
# SCANNERS (Tabs 6–10) under scan_tab
# -------------------------------------------------------
@st.cache_data(ttl=120)
def _get_close_daily(sym: str) -> pd.Series:
    return fetch_hist(sym)

@st.cache_data(ttl=120)
def _get_close_intraday(sym: str, period: str) -> pd.DataFrame:
    return fetch_intraday(sym, period=period)

def _sr_from_series(s: pd.Series, lb: int):
    if s is None or s.empty:
        return np.nan, np.nan
    res = s.rolling(int(lb), min_periods=1).max()
    sup = s.rolling(int(lb), min_periods=1).min()
    return float(sup.iloc[-1]), float(res.iloc[-1])

def _near_sr(last_px: float, sup: float, res: float, prox_pct: float) -> str:
    if not (np.isfinite(last_px) and np.isfinite(sup) and np.isfinite(res)):
        return ""
    rng = max(res - sup, 1e-12)
    tol = prox_pct * rng
    if abs(last_px - sup) <= tol:
        return "Near Support"
    if abs(last_px - res) <= tol:
        return "Near Resistance"
    return ""

with scan_tab:
    tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "6) Fib Reversal Scanner",
        "7) NTD Bottom Scanner",
        "8) Intraday Red Signal Scanner",
        "9) NTD Channel View",
        "10) Fib BUY-only Intraday",
    ])

    # ---------------------------
    # TAB 6: Fib Reversal Scanner (Historical vs last 5 days intraday)
    # ---------------------------
    with tab6:
        st.header("Fib Reversal Scanner")
        st.caption("Confirmed fib reversals: BUY from 100% (low) after reversal, SELL from 0% (high) after reversal.")

        frame = st.radio("Frame:", ["Historical (Daily)", "Last 5 days (Intraday)"], index=0, key=f"fib_scan_frame_{mode}")
        run = st.button("Run Fib Scanner", key=f"btn_run_fib_scan_{mode}")

        if run:
            rows = []
            if frame.startswith("Historical"):
                for sym in universe:
                    s = _get_close_daily(sym).dropna()
                    s = subset_by_daily_view(s, "Historical")
                    tr = fib_reversal_trigger_from_extremes(
                        s, proximity_pct_of_range=0.02, confirm_bars=rev_bars_confirm, lookback_bars=120
                    )
                    if tr is not None:
                        rows.append({
                            "Symbol": sym,
                            "Side": tr["side"],
                            "Fib Level": tr["from_level"],
                            "Time": tr["last_time"],
                            "Price": tr["last_price"],
                        })
            else:
                for sym in universe:
                    df_i = _get_close_intraday(sym, period="5d")
                    if df_i is None or df_i.empty or "Close" not in df_i:
                        continue
                    s = df_i["Close"].astype(float).dropna()
                    tr = fib_reversal_trigger_from_extremes(
                        s, proximity_pct_of_range=0.02, confirm_bars=rev_bars_confirm, lookback_bars=600
                    )
                    if tr is not None:
                        rows.append({
                            "Symbol": sym,
                            "Side": tr["side"],
                            "Fib Level": tr["from_level"],
                            "Time": tr["last_time"],
                            "Price": tr["last_price"],
                        })

            if not rows:
                st.info("No confirmed fib reversals found.")
            else:
                out = pd.DataFrame(rows).sort_values(["Side", "Time"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

                st.markdown("**Quick action**")
                pick = st.selectbox("Open this symbol in Tab 1:", out["Symbol"].tolist(), key=f"fib_scan_pick_{mode}")
                if st.button("Open in Tab 1 (loads data)", key=f"btn_open_from_fib_{mode}"):
                    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
                    _run_and_store(sel=pick, chart="Both", hour_range=st.session_state.get("hour_range", "24h"), period_map=period_map)
                    st.success(f"Loaded {pick}. Go to **Tab 1** to view.")

    # ---------------------------
    # TAB 7: NTD Bottom Scanner
    # ---------------------------
    with tab7:
        st.header("NTD Bottom Scanner")
        st.caption("Finds symbols where the latest NTD is at/near the minimum over a lookback window.")

        scan_frame = st.radio("Frame:", ["Intraday (1d)", "Intraday (5d)", "Daily"], index=0, key=f"ntd_bottom_frame_{mode}")
        bottom_lb = st.slider("Bottom lookback (bars)", 30, 800, 200, 10, key=f"ntd_bottom_lb_{mode}")
        bottom_eps = st.slider("Bottom tolerance (NTD)", 0.00, 0.20, 0.02, 0.01, key=f"ntd_bottom_eps_{mode}")

        filtered = st.checkbox("Filtered: require near S/R and NTD within a range", value=True, key=f"ntd_bottom_filtered_{mode}")
        ntd_lo = st.slider("NTD min (filtered)", -1.00, 1.00, -0.50, 0.05, key=f"ntd_bottom_ntd_lo_{mode}")
        ntd_hi = st.slider("NTD max (filtered)", -1.00, 1.00, 0.50, 0.05, key=f"ntd_bottom_ntd_hi_{mode}")

        run = st.button("Run NTD Bottom Scanner", key=f"btn_run_ntd_bottom_{mode}")

        if run:
            rows = []
            for sym in universe:
                if scan_frame == "Daily":
                    s = _get_close_daily(sym).dropna()
                    ntd = compute_normalized_trend(s, window=ntd_window).dropna()
                    if ntd.empty:
                        continue
                    ntd_tail = ntd.iloc[-bottom_lb:] if len(ntd) > bottom_lb else ntd
                    last = float(ntd_tail.iloc[-1])
                    mn = float(ntd_tail.min())
                    is_bottom = (last <= mn + float(bottom_eps))
                    last_px = float(s.iloc[-1])
                    sup, res = _sr_from_series(s, sr_lb_daily)
                else:
                    period = "1d" if scan_frame == "Intraday (1d)" else "5d"
                    df_i = _get_close_intraday(sym, period=period)
                    if df_i is None or df_i.empty or "Close" not in df_i:
                        continue
                    s = df_i["Close"].astype(float).dropna()
                    ntd = compute_normalized_trend(s, window=ntd_window).dropna()
                    if ntd.empty:
                        continue
                    ntd_tail = ntd.iloc[-bottom_lb:] if len(ntd) > bottom_lb else ntd
                    last = float(ntd_tail.iloc[-1])
                    mn = float(ntd_tail.min())
                    is_bottom = (last <= mn + float(bottom_eps))
                    last_px = float(s.iloc[-1])
                    sup, res = _sr_from_series(s, sr_lb_hourly)

                if not is_bottom:
                    continue

                near = _near_sr(last_px, sup, res, sr_prox_pct)
                if filtered:
                    if not (np.isfinite(last) and float(ntd_lo) <= last <= float(ntd_hi)):
                        continue
                    if near == "":
                        continue

                rows.append({
                    "Symbol": sym,
                    "NTD": last,
                    "Last Price": last_px,
                    "Near S/R": near,
                    "Support": sup,
                    "Resistance": res,
                })

            if not rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows).sort_values("NTD")
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

                st.markdown("**Quick action**")
                pick = st.selectbox("Open this symbol in Tab 1:", out["Symbol"].tolist(), key=f"ntd_bottom_pick_{mode}")
                if st.button("Open in Tab 1 (loads data)", key=f"btn_open_from_ntd_bottom_{mode}"):
                    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
                    _run_and_store(sel=pick, chart="Both", hour_range=st.session_state.get("hour_range", "24h"), period_map=period_map)
                    st.success(f"Loaded {pick}. Go to **Tab 1** to view.")

    # ---------------------------
    # TAB 8: Intraday Red Signal Scanner
    # ---------------------------
    with tab8:
        st.header("Intraday Red Signal Scanner")
        st.caption("A conservative 'red' condition combining intraday NTD, RSI, and Supertrend direction.")

        period = st.selectbox("Intraday period:", ["1d", "2d", "5d"], index=0, key=f"red_scan_period_{mode}")
        rsi_thr = st.slider("RSI threshold (below)", 10, 60, 40, 1, key=f"red_scan_rsi_thr_{mode}")
        ntd_thr = st.slider("NTD threshold (below)", -1.00, 0.50, -0.50, 0.05, key=f"red_scan_ntd_thr_{mode}")

        run = st.button("Run Red Signal Scanner", key=f"btn_run_red_scan_{mode}")

        if run:
            rows = []
            for sym in universe:
                df_i = _get_close_intraday(sym, period=period)
                if df_i is None or df_i.empty or not {"High", "Low", "Close"}.issubset(df_i.columns):
                    continue
                c = df_i["Close"].astype(float).dropna()
                if c.empty:
                    continue

                ntd = compute_normalized_trend(c, window=ntd_window).dropna()
                rsi = compute_rsi(c, period=14).dropna()

                if ntd.empty or rsi.empty:
                    continue

                st_line, st_dir = compute_supertrend(df_i[["High", "Low", "Close"]], period=atr_period, mult=atr_mult)
                st_dir_last = float(st_dir.dropna().iloc[-1]) if st_dir.dropna().shape[0] else np.nan

                last_ntd = float(ntd.iloc[-1])
                last_rsi = float(rsi.iloc[-1])
                last_px = float(c.iloc[-1])
                is_bear = (np.isfinite(st_dir_last) and st_dir_last < 0)

                if (last_ntd < float(ntd_thr)) and (last_rsi < float(rsi_thr)) and is_bear:
                    sup, res = _sr_from_series(c, sr_lb_hourly)
                    rows.append({
                        "Symbol": sym,
                        "NTD": last_ntd,
                        "RSI": last_rsi,
                        "Last Price": last_px,
                        "Near S/R": _near_sr(last_px, sup, res, sr_prox_pct),
                    })

            if not rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows).sort_values(["NTD", "RSI"])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

                st.markdown("**Quick action**")
                pick = st.selectbox("Open this symbol in Tab 1:", out["Symbol"].tolist(), key=f"red_scan_pick_{mode}")
                if st.button("Open in Tab 1 (loads data)", key=f"btn_open_from_red_scan_{mode}"):
                    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
                    _run_and_store(sel=pick, chart="Both", hour_range=st.session_state.get("hour_range", "24h"), period_map=period_map)
                    st.success(f"Loaded {pick}. Go to **Tab 1** to view.")

    # ---------------------------
    # TAB 9: NTD Channel View
    # ---------------------------
    with tab9:
        st.header("NTD Channel View")
        st.caption("Single-symbol viewer: highlights when price is between latest S/R while showing the NTD panel.")

        sym = st.selectbox("Symbol:", universe, key=f"ntd_channel_sym_{mode}")
        period = st.selectbox("Intraday period:", ["1d", "2d", "5d"], index=0, key=f"ntd_channel_period_{mode}")
        run = st.button("Load Channel View", key=f"btn_load_ntd_channel_{mode}")

        if run:
            df_i = _get_close_intraday(sym, period=period)
            if df_i is None or df_i.empty or "Close" not in df_i:
                st.warning("No intraday data.")
            else:
                st.session_state["_ntd_chan_df"] = df_i
                st.session_state["_ntd_chan_sym"] = sym

        if "_ntd_chan_df" in st.session_state and st.session_state.get("_ntd_chan_sym") == sym:
            df_i = st.session_state["_ntd_chan_df"]
            close = df_i["Close"].astype(float).dropna()
            if close.empty:
                st.warning("No close values.")
            else:
                sup_v, res_v = _sr_from_series(close, sr_lb_hourly)
                st.write({
                    "Support": fmt_price_val(sup_v) if np.isfinite(sup_v) else "n/a",
                    "Resistance": fmt_price_val(res_v) if np.isfinite(res_v) else "n/a",
                    "Current": fmt_price_val(float(close.iloc[-1])),
                })
                render_intraday_panel(sym, df_i)

    # ---------------------------
    # TAB 10: Fib BUY-only Intraday
    # ---------------------------
    with tab10:
        st.header("Fib BUY-only Intraday Scanner")
        st.caption("Lists only confirmed BUY fib reversals from the 100% line on intraday data.")

        period = st.selectbox("Period:", ["1d", "2d", "5d"], index=0, key=f"fib_buy_period_{mode}")
        run = st.button("Run BUY-only Fib Scanner", key=f"btn_run_fib_buy_{mode}")

        if run:
            rows = []
            for sym in universe:
                df_i = _get_close_intraday(sym, period=period)
                if df_i is None or df_i.empty or "Close" not in df_i:
                    continue
                s = df_i["Close"].astype(float).dropna()
                tr = fib_reversal_trigger_from_extremes(
                    s, proximity_pct_of_range=0.02, confirm_bars=rev_bars_confirm, lookback_bars=600
                )
                if tr is not None and tr.get("side") == "BUY":
                    rows.append({
                        "Symbol": sym,
                        "Side": tr["side"],
                        "Fib Level": tr["from_level"],
                        "Time": tr["last_time"],
                        "Price": tr["last_price"],
                    })

            if not rows:
                st.info("No BUY fib reversals found.")
            else:
                out = pd.DataFrame(rows).sort_values("Time", ascending=False)
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

                st.markdown("**Quick action**")
                pick = st.selectbox("Open this symbol in Tab 1:", out["Symbol"].tolist(), key=f"fib_buy_pick_{mode}")
                if st.button("Open in Tab 1 (loads data)", key=f"btn_open_from_fib_buy_{mode}"):
                    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
                    _run_and_store(sel=pick, chart="Both", hour_range=st.session_state.get("hour_range", "24h"), period_map=period_map)
                    st.success(f"Loaded {pick}. Go to **Tab 1** to view.")
