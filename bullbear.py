# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Support/Resistance utilities + plot overlays
# ---------------------------
def compute_support_resistance_from_ohlc(ohlc: pd.DataFrame, lookback: int = 60):
    """
    Rolling Support/Resistance using rolling Low(min) and High(max).
    Returns (support_series, resistance_series) aligned to ohlc index.
    """
    if ohlc is None or ohlc.empty:
        return (pd.Series(dtype=float), pd.Series(dtype=float))
    if not {"High", "Low"}.issubset(ohlc.columns):
        return (pd.Series(dtype=float), pd.Series(dtype=float))
    lb = max(5, int(lookback))
    low = _coerce_1d_series(ohlc["Low"])
    high = _coerce_1d_series(ohlc["High"])
    sup = low.rolling(lb, min_periods=max(3, lb // 3)).min()
    res = high.rolling(lb, min_periods=max(3, lb // 3)).max()
    return sup.reindex(ohlc.index), res.reindex(ohlc.index)

def plot_fibonacci(ax, series_like: pd.Series, label_prefix: str = "Fib"):
    """
    Draw Fibonacci levels as horizontal lines on the given axis.
    """
    fibs = fibonacci_levels(series_like)
    if not fibs:
        return
    for k, v in fibs.items():
        if not np.isfinite(v):
            continue
        ax.axhline(v, linewidth=0.9, alpha=0.45, linestyle=":")
        label_on_left(ax, v, f"{label_prefix} {k}: {fmt_price_val(v)}", fontsize=8)

def _safe_ylim(ax, y: pd.Series, pad_frac: float = 0.06):
    y = _coerce_1d_series(y).dropna()
    if y.empty:
        return
    lo = float(y.min())
    hi = float(y.max())
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if hi == lo:
        hi = lo + 1e-6
    pad = (hi - lo) * pad_frac
    ax.set_ylim(lo - pad, hi + pad)

def plot_sr(ax, sup: pd.Series, res: pd.Series, show_labels: bool = True):
    """
    Plot support/resistance lines.
    """
    s = _coerce_1d_series(sup)
    r = _coerce_1d_series(res)
    if not s.dropna().empty:
        ax.plot(s.index, s.values, linewidth=1.4, alpha=0.85, linestyle="-", label="Support")
        if show_labels:
            v = _safe_last_float(s)
            if np.isfinite(v):
                label_on_left(ax, v, f"SUP {fmt_price_val(v)}", fontsize=8)
    if not r.dropna().empty:
        ax.plot(r.index, r.values, linewidth=1.4, alpha=0.85, linestyle="-", label="Resistance")
        if show_labels:
            v = _safe_last_float(r)
            if np.isfinite(v):
                label_on_left(ax, v, f"RES {fmt_price_val(v)}", fontsize=8)

def compute_global_trendline(price: pd.Series):
    """
    Global trendline fit on FULL available series (in-sample).
    Returns: (yhat_series, slope)
    """
    s = _coerce_1d_series(price).dropna()
    if len(s) < 3:
        return pd.Series(index=price.index if isinstance(price, pd.Series) else None, dtype=float), float("nan")
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    yhat = pd.Series(m * x + b, index=s.index)
    return yhat.reindex(price.index), float(m)

def plot_global_trend(ax, yhat: pd.Series, slope: float, show: bool = True):
    if not show:
        return
    y = _coerce_1d_series(yhat).dropna()
    if y.empty:
        return
    col = "green" if np.isfinite(slope) and slope >= 0 else "red"
    ax.plot(y.index, y.values, linestyle="--", linewidth=2.2, alpha=0.75,
            color=col, label=f"Global Trend ({fmt_slope(slope)}/bar)")

def plot_bbands(ax, close: pd.Series):
    mid, upper, lower, pctb, nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
    if mid.dropna().empty:
        return
    ax.plot(mid.index, mid.values, linewidth=1.1, alpha=0.9, linestyle="-", label="BB Mid")
    ax.plot(upper.index, upper.values, linewidth=1.0, alpha=0.8, linestyle="--", label=f"BB +{bb_mult}σ")
    ax.plot(lower.index, lower.values, linewidth=1.0, alpha=0.8, linestyle="--", label=f"BB -{bb_mult}σ")
    return mid, upper, lower, pctb, nbb

def _session_markers_pst(times: pd.DatetimeIndex):
    """
    Returns session marker times (PST) for intraday axis.
    London open ~ 00:00 PST (08:00 UTC winter), NY open ~ 06:30 PST (14:30 UTC).
    These are approximations; used for visual cues.
    """
    if not isinstance(times, pd.DatetimeIndex) or times.empty:
        return []
    t0 = times.min().floor("D")
    days = pd.date_range(t0, times.max().ceil("D"), freq="D", tz=PACIFIC)
    marks = []
    for d in days:
        marks.append(d + pd.Timedelta(hours=0))               # London approx
        marks.append(d + pd.Timedelta(hours=6, minutes=30))   # NY approx
    return [m for m in marks if (m >= times.min() and m <= times.max())]

def plot_sessions_pst(ax, times: pd.DatetimeIndex):
    if not show_sessions_pst:
        return
    marks = _session_markers_pst(times)
    if not marks:
        return
    for m in marks:
        ax.axvline(m, alpha=0.22, linewidth=1.0, linestyle=":")

def _make_legend_compact(ax):
    try:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        # de-duplicate while preserving order
        seen = set()
        h2, l2 = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            h2.append(h); l2.append(l)
        ax.legend(h2, l2, loc="best", fontsize=8, frameon=False)
    except Exception:
        pass


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Daily views renderer
# ---------------------------
def render_daily_views(sel: str,
                       ohlc: pd.DataFrame,
                       close: pd.Series,
                       daily_range: str,
                       slope_lb: int,
                       sr_lb: int):
    """
    Produces:
      • Daily Price chart (with local slope, optional global trendline, S/R, optional fibs, optional BBands, optional Ichimoku)
      • Daily NTD chart (with overlays and stars)
      • Daily summary metrics dict (slopes, r2, etc.)
    """
    out = {}
    if close is None or _coerce_1d_series(close).dropna().empty:
        st.warning("No daily data.")
        return out

    # Subset
    close_show = subset_by_daily_view(close, daily_range)
    ohlc_show = None
    if ohlc is not None and not ohlc.empty:
        ohlc_show = ohlc.reindex(close.index).dropna()
        ohlc_show = subset_by_daily_view(ohlc_show, daily_range)

    # S/R
    sup_d, res_d = compute_support_resistance_from_ohlc(ohlc_show if ohlc_show is not None else ohlc, lookback=sr_lb)

    # Local slope line + R²
    yhat_local, slope_local = slope_line(close_show, lookback=int(slope_lb))
    r2_local = regression_r2(close_show, lookback=int(slope_lb))

    # Global trendline (fit on full daily close, plot only if toggle ON)
    yhat_global_full, slope_global = compute_global_trendline(close)
    yhat_global = yhat_global_full.reindex(close_show.index)

    # Trade instruction values (use S/R as entry/exit anchors when available)
    buy_val = _safe_last_float(sup_d) if not sup_d.empty else _safe_last_float(close_show)
    sell_val = _safe_last_float(res_d) if not res_d.empty else _safe_last_float(close_show)
    close_val = _safe_last_float(close_show)

    # ---------------------------
    # Daily Price chart
    # ---------------------------
    figp = plt.figure(figsize=(11, 5.2))
    axp = figp.add_subplot(111)
    axp.plot(close_show.index, close_show.values, linewidth=1.6, label=f"{sel} Close")
    style_axes(axp)

    if yhat_local.dropna().shape[0] > 0:
        col = "green" if np.isfinite(slope_local) and slope_local >= 0 else "red"
        axp.plot(yhat_local.index, yhat_local.values, linestyle="--", linewidth=2.0, alpha=0.85,
                 color=col, label=f"Local Slope ({fmt_slope(slope_local)}/bar)")

    plot_global_trend(axp, yhat_global, slope_global, show=bool(show_global_trend))

    if show_bbands:
        plot_bbands(axp, close_show)

    plot_sr(axp, sup_d, res_d, show_labels=True)

    if show_fibs:
        plot_fibonacci(axp, close_show, label_prefix="Fib")

    if show_ichi and ohlc_show is not None and not ohlc_show.empty:
        tenkan, kijun, sa, sb, chikou = ichimoku_lines(
            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
        )
        if kijun.dropna().shape[0] > 0:
            axp.plot(kijun.index, kijun.values, linewidth=1.2, alpha=0.9, label=f"Kijun({ichi_base})")

    axp.set_title(f"Daily Close — {sel}")
    _safe_ylim(axp, close_show)
    _make_legend_compact(axp)

    # instruction (daily)
    instr_daily = format_trade_instruction(
        trend_slope=slope_local,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=close_val,
        symbol=sel,
        global_trend_slope=slope_global
    )
    st.markdown(f"**Daily Instruction:** {instr_daily}")

    st.pyplot(figp, clear_figure=True)
    plt.close(figp)

    # ---------------------------
    # Daily NTD chart
    # ---------------------------
    ntd_d = compute_normalized_trend(close_show, window=ntd_window)
    npx_d = compute_normalized_price(close_show, window=ntd_window)

    fign = plt.figure(figsize=(11, 4.1))
    axn = fign.add_subplot(111)
    axn.plot(ntd_d.index, ntd_d.values, linewidth=1.6, label="NTD (Daily)")
    axn.axhline(0.0, linewidth=1.0, alpha=0.4)
    axn.axhline(0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    axn.axhline(-0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    style_axes(axn)

    if shade_ntd:
        shade_ntd_regions(axn, ntd_d)

    # trend triangles on NTD based on LOCAL slope
    overlay_ntd_triangles_by_trend(axn, ntd_d, trend_slope=slope_local, upper=0.75, lower=-0.75)

    # optional NPX overlay on NTD
    if show_npx_ntd:
        overlay_npx_on_ntd(axn, npx_d, ntd_d, mark_crosses=mark_npx_cross)

    # HMA reversal markers on NTD
    if show_hma_rev_ntd:
        hma_d = compute_hma(close_show, period=hma_period)
        overlay_hma_reversal_on_ntd(axn, close_show, hma_d, lookback=hma_rev_lb, period=hma_period, ntd=ntd_d)

    # NTD reversal stars based on S/R (daily)
    overlay_ntd_sr_reversal_stars(
        axn,
        price=close_show,
        sup=sup_d,
        res=res_d,
        trend_slope=slope_local,
        ntd=ntd_d,
        prox=sr_prox_pct,
        bars_confirm=rev_bars_confirm
    )

    axn.set_title(f"NTD (Daily) — {sel}")
    axn.set_ylim(-1.05, 1.05)
    _make_legend_compact(axn)

    st.pyplot(fign, clear_figure=True)
    plt.close(fign)

    out["slope_local_daily"] = slope_local
    out["r2_local_daily"] = r2_local
    out["slope_global_daily"] = slope_global
    out["sup_daily"] = sup_d
    out["res_daily"] = res_d
    out["close_show"] = close_show
    out["ntd_daily"] = ntd_d
    return out


# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Hourly views renderer (intraday 5m aggregated to hourly)
# ---------------------------
def _resample_to_hourly(ohlc_5m: pd.DataFrame) -> pd.DataFrame:
    if ohlc_5m is None or ohlc_5m.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df = ohlc_5m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    o = df["Open"].resample("1H").first() if "Open" in df.columns else None
    h = df["High"].resample("1H").max() if "High" in df.columns else None
    l = df["Low"].resample("1H").min() if "Low" in df.columns else None
    c = df["Close"].resample("1H").last() if "Close" in df.columns else None
    v = df["Volume"].resample("1H").sum() if "Volume" in df.columns else None
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    if v is not None:
        out["Volume"] = v
    out = out.dropna(subset=["Close"])
    return out

def render_hourly_views(sel: str,
                        intraday_5m: pd.DataFrame,
                        slope_lb: int,
                        sr_lb: int,
                        is_forex: bool = False):
    """
    Produces:
      • Hourly Close chart (with local slope, optional global trendline, S/R, optional fibs, BBands, supertrend, psar)
      • Hourly NTD chart (+ overlays, including the fixed overlay_ntd_sr_reversal_stars)
      • Optional hourly momentum chart
      • Optional MACD chart (normalized)
      • Metrics dict
    """
    out = {}
    if intraday_5m is None or intraday_5m.empty or "Close" not in intraday_5m.columns:
        st.warning("No intraday data.")
        return out

    # Convert to hourly bars
    ohlc_h = _resample_to_hourly(intraday_5m)
    if ohlc_h.empty:
        st.warning("No hourly bars after resample.")
        return out

    close_h = _coerce_1d_series(ohlc_h["Close"]).dropna()
    if close_h.empty:
        st.warning("Hourly close is empty.")
        return out

    # S/R (hourly)
    sup_h, res_h = compute_support_resistance_from_ohlc(ohlc_h, lookback=sr_lb)

    # Local slope
    yhat_local, slope_local = slope_line(close_h, lookback=int(slope_lb))
    r2_local = regression_r2(close_h, lookback=int(slope_lb))

    # Global trendline computed on hourly visible series for stability (intraday only)
    yhat_global, slope_global = compute_global_trendline(close_h)

    buy_val = _safe_last_float(sup_h) if not sup_h.empty else _safe_last_float(close_h)
    sell_val = _safe_last_float(res_h) if not res_h.empty else _safe_last_float(close_h)
    close_val = _safe_last_float(close_h)

    # ---------------------------
    # Hourly Price chart
    # ---------------------------
    figp = plt.figure(figsize=(11, 5.2))
    axp = figp.add_subplot(111)
    axp.plot(close_h.index, close_h.values, linewidth=1.6, label=f"{sel} Hourly Close")
    style_axes(axp)
    plot_sessions_pst(axp, close_h.index)

    if yhat_local.dropna().shape[0] > 0:
        col = "green" if np.isfinite(slope_local) and slope_local >= 0 else "red"
        axp.plot(yhat_local.index, yhat_local.values, linestyle="--", linewidth=2.0, alpha=0.85,
                 color=col, label=f"Local Slope ({fmt_slope(slope_local)}/bar)")

    plot_global_trend(axp, yhat_global, slope_global, show=bool(show_global_trend))

    if show_bbands:
        plot_bbands(axp, close_h)

    plot_sr(axp, sup_h, res_h, show_labels=True)

    if show_fibs:
        plot_fibonacci(axp, close_h, label_prefix="Fib")

    # Supertrend
    try:
        st_df = compute_supertrend(ohlc_h, atr_period=atr_period, atr_mult=atr_mult)
        if not st_df.empty and "ST" in st_df.columns:
            axp.plot(st_df.index, st_df["ST"].values, linewidth=1.3, alpha=0.85, label="Supertrend")
    except Exception:
        pass

    # PSAR
    if show_psar:
        try:
            ps_df = compute_psar_from_ohlc(ohlc_h, step=psar_step, max_step=psar_max)
            if not ps_df.empty and "PSAR" in ps_df.columns:
                axp.scatter(ps_df.index, ps_df["PSAR"].values, s=12, alpha=0.65, label="PSAR")
        except Exception:
            pass

    axp.set_title(f"Hourly Close — {sel}")
    _safe_ylim(axp, close_h)
    _make_legend_compact(axp)

    instr_hourly = format_trade_instruction(
        trend_slope=slope_local,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=close_val,
        symbol=sel,
        global_trend_slope=slope_global
    )
    st.markdown(f"**Hourly Instruction:** {instr_hourly}")

    st.pyplot(figp, clear_figure=True)
    plt.close(figp)

    # ---------------------------
    # Hourly NTD chart
    # ---------------------------
    ntd_h = compute_normalized_trend(close_h, window=ntd_window)
    npx_h = compute_normalized_price(close_h, window=ntd_window)

    fign = plt.figure(figsize=(11, 4.1))
    axn = fign.add_subplot(111)
    axn.plot(ntd_h.index, ntd_h.values, linewidth=1.6, label="NTD (Hourly)")
    axn.axhline(0.0, linewidth=1.0, alpha=0.4)
    axn.axhline(0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    axn.axhline(-0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    style_axes(axn)
    plot_sessions_pst(axn, ntd_h.index)

    if shade_ntd:
        shade_ntd_regions(axn, ntd_h)

    overlay_ntd_triangles_by_trend(axn, ntd_h, trend_slope=slope_local, upper=0.75, lower=-0.75)

    if show_npx_ntd:
        overlay_npx_on_ntd(axn, npx_h, ntd_h, mark_crosses=mark_npx_cross)

    if show_hma_rev_ntd:
        hma_h = compute_hma(close_h, period=hma_period)
        overlay_hma_reversal_on_ntd(axn, close_h, hma_h, lookback=hma_rev_lb, period=hma_period, ntd=ntd_h)

    # FIXED: this function now exists (prevents NameError)
    overlay_ntd_sr_reversal_stars(
        axn,
        price=close_h,
        sup=sup_h,
        res=res_h,
        trend_slope=slope_local,
        ntd=ntd_h,
        prox=sr_prox_pct,
        bars_confirm=rev_bars_confirm
    )

    axn.set_title(f"NTD (Hourly) — {sel}")
    axn.set_ylim(-1.05, 1.05)
    _make_legend_compact(axn)

    st.pyplot(fign, clear_figure=True)
    plt.close(fign)

    # ---------------------------
    # Optional: Hourly Momentum (ROC%)
    # ---------------------------
    if show_mom_hourly:
        roc = compute_roc(close_h, n=mom_lb_hourly)
        figm = plt.figure(figsize=(11, 3.4))
        axm = figm.add_subplot(111)
        axm.plot(roc.index, roc.values, linewidth=1.4, label=f"ROC% ({mom_lb_hourly})")
        axm.axhline(0.0, linewidth=1.0, alpha=0.4)
        style_axes(axm)
        plot_sessions_pst(axm, roc.index)
        axm.set_title(f"Hourly Momentum — {sel}")
        _make_legend_compact(axm)
        st.pyplot(figm, clear_figure=True)
        plt.close(figm)

    # ---------------------------
    # Optional: MACD (Normalized)
    # ---------------------------
    if show_macd:
        nmacd, nsig, nhist = compute_nmacd(close_h, fast=12, slow=26, signal=9, norm_win=240)
        figc = plt.figure(figsize=(11, 3.6))
        axc = figc.add_subplot(111)
        axc.plot(nmacd.index, nmacd.values, linewidth=1.3, label="N-MACD")
        axc.plot(nsig.index, nsig.values, linewidth=1.1, alpha=0.9, label="N-Signal")
        axc.axhline(0.0, linewidth=1.0, alpha=0.35)
        style_axes(axc)
        plot_sessions_pst(axc, nmacd.index)
        axc.set_ylim(-1.05, 1.05)
        axc.set_title(f"Normalized MACD — {sel}")
        _make_legend_compact(axc)
        st.pyplot(figc, clear_figure=True)
        plt.close(figc)

    out["slope_local_hourly"] = slope_local
    out["r2_local_hourly"] = r2_local
    out["slope_global_hourly"] = slope_global
    out["sup_hourly"] = sup_h
    out["res_hourly"] = res_h
    out["close_hourly"] = close_h
    out["ntd_hourly"] = ntd_h
    out["ohlc_hourly"] = ohlc_h
    return out


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Scanners (simple + fast; keeps UI consistent)
# ---------------------------
@st.cache_data(ttl=120)
def _scan_one_ticker_daily(ticker: str):
    try:
        ohlc = fetch_hist_ohlc(ticker)
        close = ohlc["Close"].asfreq("D").ffill()
        close = close.tz_convert(PACIFIC) if close.index.tz is not None else close.tz_localize(PACIFIC)
    except Exception:
        return None

    close_show = subset_by_daily_view(close, daily_view)
    yhat_local, slope_local = slope_line(close_show, lookback=int(slope_lb_daily))
    r2 = regression_r2(close_show, lookback=int(slope_lb_daily))
    yhat_g, slope_g = compute_global_trendline(close)

    ntd = compute_normalized_trend(close_show, window=ntd_window)
    ntd_last = _safe_last_float(ntd)

    sup, res = compute_support_resistance_from_ohlc(ohlc, lookback=sr_lb_daily)
    sup_last = _safe_last_float(sup)
    res_last = _safe_last_float(res)
    px_last = _safe_last_float(close_show)

    fib_trig = fib_reversal_trigger_from_extremes(close_show, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=60)

    return {
        "Ticker": ticker,
        "Last": px_last,
        "LocalSlope": slope_local,
        "GlobalSlope": slope_g,
        "R2": r2,
        "NTD": ntd_last,
        "SUP": sup_last,
        "RES": res_last,
        "FibTrig": (fib_trig["side"] if fib_trig else ""),
    }

@st.cache_data(ttl=120)
def _scan_one_ticker_intraday(ticker: str):
    try:
        intra = fetch_intraday(ticker, period="5d")
        if intra is None or intra.empty or "Close" not in intra.columns:
            return None
        ohlc_h = _resample_to_hourly(intra)
        if ohlc_h.empty:
            return None
        close_h = _coerce_1d_series(ohlc_h["Close"]).dropna()
    except Exception:
        return None

    yhat_local, slope_local = slope_line(close_h, lookback=int(slope_lb_hourly))
    r2 = regression_r2(close_h, lookback=int(slope_lb_hourly))
    yhat_g, slope_g = compute_global_trendline(close_h)

    ntd = compute_normalized_trend(close_h, window=ntd_window)
    ntd_last = _safe_last_float(ntd)

    sup, res = compute_support_resistance_from_ohlc(ohlc_h, lookback=sr_lb_hourly)
    sup_last = _safe_last_float(sup)
    res_last = _safe_last_float(res)
    px_last = _safe_last_float(close_h)

    fib_trig = fib_reversal_trigger_from_extremes(close_h, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=80)

    return {
        "Ticker": ticker,
        "Last": px_last,
        "LocalSlope": slope_local,
        "GlobalSlope": slope_g,
        "R2": r2,
        "NTD": ntd_last,
        "SUP": sup_last,
        "RES": res_last,
        "FibTrig": (fib_trig["side"] if fib_trig else ""),
    }

def render_scanners(universe_list):
    st.subheader("Scanners")

    s_tabs = st.tabs(["Daily Trend", "Hourly Trend", "Fib Reversal", "MACD/HMA + S/R"])

    with s_tabs[0]:
        st.caption("Daily snapshot across the current universe.")
        rows = []
        for t in universe_list:
            r = _scan_one_ticker_daily(t)
            if r:
                rows.append(r)
        if not rows:
            st.info("No scan results.")
        else:
            df = pd.DataFrame(rows)
            # light ranking: prefer agreement of local+global slope and strong fit
            df["Agree"] = np.sign(df["LocalSlope"].astype(float)) == np.sign(df["GlobalSlope"].astype(float))
            df = df.sort_values(["Agree", "R2"], ascending=[False, False])
            st.dataframe(df, use_container_width=True, height=420)

    with s_tabs[1]:
        st.caption("Hourly snapshot (uses 5-day intraday data).")
        rows = []
        for t in universe_list:
            r = _scan_one_ticker_intraday(t)
            if r:
                rows.append(r)
        if not rows:
            st.info("No scan results.")
        else:
            df = pd.DataFrame(rows)
            df["Agree"] = np.sign(df["LocalSlope"].astype(float)) == np.sign(df["GlobalSlope"].astype(float))
            df = df.sort_values(["Agree", "R2"], ascending=[False, False])
            st.dataframe(df, use_container_width=True, height=420)

    with s_tabs[2]:
        st.caption("Confirmed fib reversal signals (0% / 100%) using the confirmation rule.")
        st.info(FIB_ALERT_TEXT)
        rows_d, rows_h = [], []
        for t in universe_list:
            rd = _scan_one_ticker_daily(t)
            rh = _scan_one_ticker_intraday(t)
            if rd and rd.get("FibTrig"):
                rows_d.append(rd)
            if rh and rh.get("FibTrig"):
                rows_h.append(rh)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Daily Confirmed**")
            if rows_d:
                st.dataframe(pd.DataFrame(rows_d).sort_values(["FibTrig","R2"], ascending=[True, False]),
                             use_container_width=True, height=360)
            else:
                st.write("None.")
        with c2:
            st.markdown("**Hourly Confirmed**")
            if rows_h:
                st.dataframe(pd.DataFrame(rows_h).sort_values(["FibTrig","R2"], ascending=[True, False]),
                             use_container_width=True, height=360)
            else:
                st.write("None.")

    with s_tabs[3]:
        st.caption("Signal occurs when MACD/HMA55 cross aligns with S/R proximity and Global Trendline direction.")
        rows = []
        for t in universe_list:
            try:
                intra = fetch_intraday(t, period="5d")
                ohlc_h = _resample_to_hourly(intra)
                if ohlc_h.empty:
                    continue
                close_h = _coerce_1d_series(ohlc_h["Close"]).dropna()
                sup, res = compute_support_resistance_from_ohlc(ohlc_h, lookback=sr_lb_hourly)
                hma = compute_hma(close_h, period=hma_period)
                macd, sig, hist = compute_macd(close_h)
                yhat_g, slope_g = compute_global_trendline(close_h)
                sigd = find_macd_hma_sr_signal(close_h, hma, macd, sup, res, global_trend_slope=slope_g, prox=sr_prox_pct)
                if sigd:
                    rows.append({
                        "Ticker": t,
                        "Side": sigd.get("side", ""),
                        "Time": sigd.get("time", ""),
                        "Price": sigd.get("price", np.nan),
                        "GlobalSlope": slope_g
                    })
            except Exception:
                continue

        if not rows:
            st.write("No signals found.")
        else:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Main app execution
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False

# Ticker picker (robust to mode switches)
default_ticker = universe[0] if universe else None
if st.session_state.get("ticker") not in universe:
    st.session_state.ticker = default_ticker

disp_ticker = st.selectbox(
    "Select ticker:",
    options=universe,
    index=universe.index(st.session_state.ticker) if st.session_state.ticker in universe else 0,
    key="ticker_selectbox"
)
st.session_state.ticker = disp_ticker

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    if st.button("▶ Run", use_container_width=True, key="btn_run"):
        st.session_state.run_all = True
        st.session_state.mode_at_run = mode

with run_col2:
    st.caption("Tip: Switching Forex/Stocks resets run state to prevent stale selects.")

if not st.session_state.run_all:
    st.info("Select a ticker and click **Run**.")
    st.stop()

# Guard: mode changed after running
if st.session_state.get("mode_at_run") != mode:
    st.warning("Mode changed since last run. Please click **Run** again.")
    st.stop()

# ---------------------------
# Load data
# ---------------------------
with st.spinner("Loading data…"):
    df_ohlc = fetch_hist_ohlc(disp_ticker)
    df_close = df_ohlc["Close"].asfreq("D").ffill()
    try:
        df_close = df_close.tz_convert(PACIFIC) if df_close.index.tz is not None else df_close.tz_localize(PACIFIC)
    except Exception:
        pass

    # Intraday (5d gives enough hourly bars)
    df_intra = fetch_intraday(disp_ticker, period="5d")

# ---------------------------
# Forecast (Daily SARIMAX)
# ---------------------------
fc_idx, fc_vals, fc_ci = (None, None, None)
try:
    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_close)
except Exception:
    pass

# ---------------------------
# Tabs (10 total)
# ---------------------------
tabs = st.tabs([
    "1) Daily Close",
    "2) Hourly Close",
    "3) NTD Daily",
    "4) NTD Hourly",
    "5) Forecast",
    "6) MACD",
    "7) Momentum",
    "8) 2σ Reversal",
    "9) Fib 0/100 Reversal",
    "10) Scanners"
])

# Precompute render outputs (so each tab is fast + consistent)
daily_out = {}
hourly_out = {}

with tabs[0]:
    st.subheader("Daily Close")
    daily_out = render_daily_views(
        sel=disp_ticker,
        ohlc=df_ohlc,
        close=df_close,
        daily_range=daily_view,
        slope_lb=slope_lb_daily,
        sr_lb=sr_lb_daily
    )

with tabs[1]:
    st.subheader("Hourly Close")
    hourly_out = render_hourly_views(
        sel=disp_ticker,
        intraday_5m=df_intra,
        slope_lb=slope_lb_hourly,
        sr_lb=sr_lb_hourly,
        is_forex=(mode == "Forex")
    )

with tabs[2]:
    st.subheader("NTD Daily")
    if daily_out and "ntd_daily" in daily_out:
        # Daily NTD already shown in render_daily_views, but keep this tab as a dedicated display (no UI change)
        st.info("Daily NTD is shown in the Daily section above. This tab is kept for consistent navigation.")
    else:
        st.warning("Run Daily view first.")

with tabs[3]:
    st.subheader("NTD Hourly")
    if hourly_out and "ntd_hourly" in hourly_out:
        st.info("Hourly NTD is shown in the Hourly section above. This tab is kept for consistent navigation.")
    else:
        st.warning("Run Hourly view first.")

with tabs[4]:
    st.subheader("Forecast (Daily SARIMAX)")
    if fc_idx is None or fc_vals is None or len(fc_vals) == 0:
        st.warning("Forecast unavailable.")
    else:
        figf = plt.figure(figsize=(11, 4.8))
        axf = figf.add_subplot(111)
        show = subset_by_daily_view(df_close, daily_view)
        axf.plot(show.index, show.values, linewidth=1.6, label="History")
        axf.plot(fc_idx, fc_vals.values, linewidth=1.6, linestyle="--", label="Forecast")
        try:
            lo = fc_ci.iloc[:, 0].values
            hi = fc_ci.iloc[:, 1].values
            axf.fill_between(fc_idx, lo, hi, alpha=0.18)
        except Exception:
            pass
        style_axes(axf)
        axf.set_title(f"30-Day Forecast — {disp_ticker}")
        _make_legend_compact(axf)
        st.pyplot(figf, clear_figure=True)
        plt.close(figf)

with tabs[5]:
    st.subheader("MACD (Daily + Hourly if available)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Daily MACD**")
        close_show = subset_by_daily_view(df_close, daily_view)
        macd_d, sig_d, hist_d = compute_macd(close_show)
        fig = plt.figure(figsize=(9, 3.6))
        ax = fig.add_subplot(111)
        ax.plot(macd_d.index, macd_d.values, linewidth=1.3, label="MACD")
        ax.plot(sig_d.index, sig_d.values, linewidth=1.1, alpha=0.9, label="Signal")
        ax.axhline(0.0, linewidth=1.0, alpha=0.35)
        style_axes(ax)
        ax.set_title("Daily MACD")
        _make_legend_compact(ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with c2:
        st.markdown("**Hourly MACD**")
        if hourly_out and "close_hourly" in hourly_out:
            close_h = hourly_out["close_hourly"]
            macd_h, sig_h, hist_h = compute_macd(close_h)
            fig = plt.figure(figsize=(9, 3.6))
            ax = fig.add_subplot(111)
            ax.plot(macd_h.index, macd_h.values, linewidth=1.3, label="MACD")
            ax.plot(sig_h.index, sig_h.values, linewidth=1.1, alpha=0.9, label="Signal")
            ax.axhline(0.0, linewidth=1.0, alpha=0.35)
            style_axes(ax)
            plot_sessions_pst(ax, macd_h.index)
            ax.set_title("Hourly MACD")
            _make_legend_compact(ax)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.write("Run Hourly view first.")

with tabs[6]:
    st.subheader("Momentum (ROC%)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Daily ROC%**")
        close_show = subset_by_daily_view(df_close, daily_view)
        roc_d = compute_roc(close_show, n=10)
        fig = plt.figure(figsize=(9, 3.4))
        ax = fig.add_subplot(111)
        ax.plot(roc_d.index, roc_d.values, linewidth=1.4, label="ROC% (10)")
        ax.axhline(0.0, linewidth=1.0, alpha=0.4)
        style_axes(ax)
        ax.set_title("Daily Momentum")
        _make_legend_compact(ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with c2:
        st.markdown("**Hourly ROC%**")
        if hourly_out and "close_hourly" in hourly_out:
            close_h = hourly_out["close_hourly"]
            roc_h = compute_roc(close_h, n=mom_lb_hourly)
            fig = plt.figure(figsize=(9, 3.4))
            ax = fig.add_subplot(111)
            ax.plot(roc_h.index, roc_h.values, linewidth=1.4, label=f"ROC% ({mom_lb_hourly})")
            ax.axhline(0.0, linewidth=1.0, alpha=0.4)
            style_axes(ax)
            plot_sessions_pst(ax, roc_h.index)
            ax.set_title("Hourly Momentum")
            _make_legend_compact(ax)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.write("Run Hourly view first.")

with tabs[7]:
    st.subheader("2σ Reversal (Bands + Slope Trigger)")
    st.caption("This tab uses linear regression ±2σ bands and slope-cross triggers.")

    close_show = subset_by_daily_view(df_close, daily_view)
    yhat, upper, lower, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)

    # Reversal probability (experimental)
    prob = slope_reversal_probability(
        close_show,
        current_slope=m,
        hist_window=rev_hist_lb,
        slope_window=ntd_window,
        horizon=rev_horizon
    )

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(close_show.index, close_show.values, linewidth=1.6, label="Price")
    if yhat.dropna().shape[0] > 0:
        ax.plot(yhat.index, yhat.values, linewidth=2.0, linestyle="--", alpha=0.9, label="Slope Line")
    if upper.dropna().shape[0] > 0 and lower.dropna().shape[0] > 0:
        ax.plot(upper.index, upper.values, linewidth=1.0, linestyle="--", alpha=0.7, label="+2σ")
        ax.plot(lower.index, lower.values, linewidth=1.0, linestyle="--", alpha=0.7, label="-2σ")

    style_axes(ax)
    ax.set_title(f"2σ Bands + Slope Trigger — {disp_ticker}")
    _safe_ylim(ax, close_show)

    # Band bounce signal
    bounce = find_band_bounce_signal(close_show, upper, lower, slope_val=m)
    if bounce:
        annotate_crossover(ax, bounce["time"], bounce["price"], bounce["side"], note="(2σ bounce)")

    # Slope trigger after band reversal
    trig = find_slope_trigger_after_band_reversal(close_show, yhat, upper, lower, horizon=rev_horizon)
    annotate_slope_trigger(ax, trig)

    _make_legend_compact(ax)

    st.metric("Reversal probability (experimental)", fmt_pct(prob, digits=1) if np.isfinite(prob) else "n/a")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with tabs[8]:
    st.subheader("Fib 0/100 Reversal (Confirmed)")
    st.info(FIB_ALERT_TEXT)

    close_show = subset_by_daily_view(df_close, daily_view)
    trig_d = fib_reversal_trigger_from_extremes(close_show, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=60)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Daily Confirmation**")
        if trig_d is None:
            st.write("No confirmed fib reversal (daily).")
        else:
            st.success(f"Confirmed **{trig_d['side']}** from **{trig_d['from_level']}** "
                       f"(touch: {trig_d['touch_time']}, last: {trig_d['last_time']})")

        fig = plt.figure(figsize=(9.2, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(close_show.index, close_show.values, linewidth=1.6, label="Daily Close")
        style_axes(ax)
        if show_fibs:
            plot_fibonacci(ax, close_show, label_prefix="Fib")
        ax.set_title(f"Daily Fib Levels — {disp_ticker}")
        _make_legend_compact(ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with c2:
        st.markdown("**Hourly Confirmation**")
        if hourly_out and "close_hourly" in hourly_out:
            close_h = hourly_out["close_hourly"]
            trig_h = fib_reversal_trigger_from_extremes(close_h, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=80)
            if trig_h is None:
                st.write("No confirmed fib reversal (hourly).")
            else:
                st.success(f"Confirmed **{trig_h['side']}** from **{trig_h['from_level']}** "
                           f"(touch: {trig_h['touch_time']}, last: {trig_h['last_time']})")

            fig = plt.figure(figsize=(9.2, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(close_h.index, close_h.values, linewidth=1.6, label="Hourly Close")
            style_axes(ax)
            plot_sessions_pst(ax, close_h.index)
            if show_fibs:
                plot_fibonacci(ax, close_h, label_prefix="Fib")
            ax.set_title(f"Hourly Fib Levels — {disp_ticker}")
            _make_legend_compact(ax)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.write("Run Hourly view first.")

with tabs[9]:
    render_scanners(universe)

# Footer note
st.caption("Signals are informational only. Use risk management and confirm with your own analysis.")
