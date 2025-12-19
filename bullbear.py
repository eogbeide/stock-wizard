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

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Upward Slope Stickers",
    "Daily Support Reversals"
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
            # KEEP BUY Band REV (triangle + badge). REMOVE SELL Band REV callout.
            if band_sig_d is not None and band_sig_d.get("side") == "BUY":
                badges_top.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_d['price'])}", "tab:green"))
                annotate_buy_triangle(ax, band_sig_d["time"], band_sig_d["price"])

            # Star (Daily)
            star_d = last_reversal_star(df_show, trend_slope=m_d, lookback=20, confirm_bars=rev_bars_confirm)
            if star_d is not None:
                annotate_star(ax, star_d["time"], star_d["price"], star_d["kind"], show_text=False)
                if star_d.get("kind") == "trough":
                    badges_top.append((f"★ Trough REV @{fmt_price_val(star_d['price'])}", "tab:green"))
                elif star_d.get("kind") == "peak":
                    badges_top.append((f"★ Peak REV @{fmt_price_val(star_d['price'])}", "tab:red"))

            # HMA-cross star (Daily)
            hma_cross_star = last_hma_cross_star(df_show, hma_d_full, trend_slope=m_d, lookback=30)
            if hma_cross_star is not None:
                if hma_cross_star["kind"] == "trough":
                    annotate_star(ax, hma_cross_star["time"], hma_cross_star["price"], hma_cross_star["kind"], show_text=False, color_override="black")
                    badges_top.append((f"★ Buy HMA Cross @{fmt_price_val(hma_cross_star['price'])}", "black"))
                else:
                    annotate_star(ax, hma_cross_star["time"], hma_cross_star["price"], hma_cross_star["kind"], show_text=False, color_override="tab:blue")
                    badges_top.append((f"★ Sell HMA Cross @{fmt_price_val(hma_cross_star['price'])}", "tab:blue"))

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
                if sr99_sig["side"] == "BUY":
                    annotate_signal_box(ax, sr99_sig["time"], sr99_sig["price"], side="BUY", note=sr99_sig["note"])
                    badges_top.append((f"▲ BUY ALERT 99% SR REV @{fmt_price_val(sr99_sig['price'])}", "tab:green"))
                else:
                    annotate_signal_box(ax, sr99_sig["time"], sr99_sig["price"], side="SELL", note=sr99_sig["note"])
                    badges_top.append((f"▼ SELL ALERT 99% SR REV @{fmt_price_val(sr99_sig['price'])}", "tab:red"))

            draw_top_badges(ax, badges_top)

            # --- NEW (Addition only): MACD normalized-to-price overlay (Daily) ---
            if show_macd:
                try:
                    macd_line_d, macd_sig_d, _ = compute_macd(df_show, fast=macd_fast, slow=macd_slow, signal=macd_signal)
                    macd_ratio_d = (macd_line_d / _coerce_1d_series(df_show).replace(0, np.nan))
                    max_abs_d = float(np.nanmax(np.abs(macd_ratio_d.to_numpy(dtype=float)))) if len(macd_ratio_d) else np.nan
                    if np.isfinite(max_abs_d) and max_abs_d > 0:
                        ymin, ymax = ax.get_ylim()
                        pr = (ymax - ymin) if (np.isfinite(ymax) and np.isfinite(ymin)) else 1.0
                        base = ymin + pr * 0.12
                        amp  = pr * 0.10
                        macd_overlay_d = base + (macd_ratio_d / max_abs_d) * amp
                        ax.plot(df_show.index, macd_overlay_d.values, "-", linewidth=1.05, alpha=0.75, label="MACD (norm)")
                        ax.hlines(base, xmin=df_show.index[0], xmax=df_show.index[-1],
                                  colors="grey", linestyles="--", linewidth=0.7, alpha=0.25, label="_nolegend_")
                except Exception:
                    pass

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
                hma_h = compute_hma(hc, period=hma_period)

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

                if show_hma and not hma_h.dropna().empty:
                    ax2.plot(x_mt, hma_h.reindex(idx_mt).values, "-", linewidth=1.3, alpha=0.9, label="HMA")

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
                # KEEP BUY Band REV only. REMOVE SELL Band REV callout.
                if band_sig_h is not None:
                    tpos = _pos(band_sig_h["time"])
                    if np.isfinite(tpos) and band_sig_h.get("side") == "BUY":
                        badges_top_h.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_h['price'])}", "tab:green"))
                        annotate_buy_triangle(ax2, tpos, band_sig_h["price"])

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

                draw_top_badges(ax2, badges_top_h)

                # TOP instruction banner (Hourly) — LOCAL=dashed slope_h; GLOBAL=m_h
                draw_instruction_ribbons(ax2, slope_h, sup_val, res_val, px_val, sel,
                                         confirm_side=None,
                                         global_slope=m_h)

                if np.isfinite(px_val):
                    nbb_txt = ""
                    try:
                        last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        if np.isfinite(last_nbb) and np.isfinite(last_pct):
                            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                    except Exception:
                        pass
                    footer_txt = (
                        f"Current price: {fmt_price_val(px_val)}{nbb_txt}\n"
                        f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}  •  Slope: {fmt_slope(m_h)}/bar"
                    )
                    ax2.text(0.5, 0.02, footer_txt,
                             transform=ax2.transAxes, ha="center", va="bottom",
                             fontsize=10, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if not st_line_intr.dropna().empty:
                    ax2.plot(x_mt, st_line_intr.reindex(idx_mt).values, "-", alpha=0.6, label="_nolegend_")
                if not yhat_h.empty:
                    slope_col_h = "tab:green" if m_h >= 0 else "tab:red"
                    ax2.plot(x_mt, yhat_h.reindex(idx_mt).values, "-", linewidth=2.6, color=slope_col_h, alpha=0.95, label="Slope Fit")
                if not upper_h.empty and not lower_h.empty:
                    ax2.plot(x_mt, upper_h.reindex(idx_mt).values, ":", linewidth=2.5, color="black", alpha=1.0, label="_nolegend_")
                    ax2.plot(x_mt, lower_h.reindex(idx_mt).values, ":", linewidth=2.5, color="black", alpha=1.0, label="_nolegend_")

                if mode == "Forex" and show_sessions_pst and not hc.empty:
                    sess_dt = compute_session_lines(idx_mt)
                    sess_pos = map_session_lines_to_positions(sess_dt, idx_mt)
                    draw_session_lines(ax2, sess_pos)

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=0, xmax=len(x_mt)-1, linestyles="dotted", linewidth=0.6, alpha=0.35)
                    for lbl, y in fibs_h.items():
                        ax2.text(len(x_mt)-1, y, f" {lbl}", va="center", fontsize=8, alpha=0.6, fontweight="bold")
                    try:
                        fib0 = fibs_h.get("0%")
                        fib100 = fibs_h.get("100%")
                        if np.isfinite(fib0) and np.isfinite(fib100):
                            fib_sig = last_fib_extreme_reversal(
                                price=hc,
                                slope=slope_h,
                                fib0_level=float(fib0),
                                fib100_level=float(fib100),
                                prox=sr_prox_pct,
                                confirm_bars=rev_bars_confirm,
                                lookback=max(20, int(sr_lb_hourly))
                            )
                            if fib_sig is not None:
                                if fib_sig["dir"] == "DOWN":
                                    annotate_fib_reversal(ax2, ts=len(x_mt)-1, y_level=float(fib0), direction="DOWN", label="Fib 0% REV → 100%")
                                elif fib_sig["dir"] == "UP":
                                    annotate_fib_reversal(ax2, ts=len(x_mt)-1, y_level=float(fib100), direction="UP", label="Fib 100% REV → 0%")
                    except Exception:
                        pass

                # --- NEW (Addition only): MACD normalized-to-price overlay (Hourly) ---
                if show_macd:
                    try:
                        macd_line_h, macd_sig_h, _ = compute_macd(hc, fast=macd_fast, slow=macd_slow, signal=macd_signal)
                        macd_line_h = macd_line_h.reindex(idx_mt)
                        macd_ratio_h = (macd_line_h / hc.replace(0, np.nan))
                        max_abs_h = float(np.nanmax(np.abs(macd_ratio_h.to_numpy(dtype=float)))) if len(macd_ratio_h) else np.nan
                        if np.isfinite(max_abs_h) and max_abs_h > 0:
                            ymin, ymax = ax2.get_ylim()
                            pr = (ymax - ymin) if (np.isfinite(ymax) and np.isfinite(ymin)) else 1.0
                            base = ymin + pr * 0.12
                            amp  = pr * 0.10
                            macd_overlay_h = base + (macd_ratio_h / max_abs_h) * amp
                            ax2.plot(x_mt, macd_overlay_h.values, "-", linewidth=1.05, alpha=0.75, label="MACD (norm)")
                            ax2.hlines(base, xmin=0, xmax=len(x_mt)-1,
                                       colors="grey", linestyles="--", linewidth=0.7, alpha=0.25, label="_nolegend_")
                    except Exception:
                        pass

                market_time_axis(ax2, idx_mt)
                _simplify_axes(ax2)
                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.4)
                pad_right_xaxis(ax2, frac=0.06)
                st.pyplot(fig2)

        if mode == "Forex":
            st.subheader("Forex Session Overlaps (PST)")
            st.markdown("""
| Overlap | Time (PST) | Applies To |
|---|---|---|
| **New York & London** | **5:00 AM – 8:00 AM** | Any pair including **EUR**, **USD**, **GBP** |
| **Tokyo & New York** | **4:00 PM – 7:00 PM** | Any pair including **USD**, **JPY** |
| **London & Tokyo** | **12:00 AM – 1:00 AM** | Any pair including **EUR**, **GBP**, **JPY** |
""")
            st.caption("These windows often see higher liquidity and volatility.")

        st.write(pd.DataFrame({"Forecast": st.session_state.fc_vals,
                               "Lower":    st.session_state.fc_ci.iloc[:,0],
                               "Upper":    st.session_state.fc_ci.iloc[:,1]}, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci)
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}**")

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        if view in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(df, slope_lb_daily, z=Z_FOR_99)

            kijun_d2 = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d2, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                      conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d2 = kijun_d2.ffill().bfill()

            bb_mid_d2, bb_up_d2, bb_lo_d2 = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)[:3]

            psar_d2_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()

            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = ema30.reindex(df_show.index)
            res30_show = res30.reindex(df_show.index)
            sup30_show = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            up_d_show   = up_d.reindex(df_show.index) if not up_d.empty else up_d
            lo_d_show   = lo_d.reindex(df_show.index) if not lo_d.empty else lo_d
            kijun_d2_show = kijun_d2.reindex(df_show.index).ffill().bfill()
            bb_mid_d2_show = bb_mid_d2.reindex(df_show.index)
            bb_up_d2_show  = bb_up_d2.reindex(df_show.index)
            bb_lo_d2_show  = bb_lo_d2.reindex(df_show.index)
            psar_d2_show = _align_series_to_index(psar_d2_df["PSAR"], df_show.index) if (show_psar and not psar_d2_df.empty and "PSAR" in psar_d2_df) else pd.Series(index=df_show.index, dtype=float)

            hma_d2_full = compute_hma(df, period=hma_period).reindex(df_show.index)

            fig, ax = plt.subplots(figsize=(14, 6))
            plt.subplots_adjust(top=0.86, right=0.995, left=0.06)

            ax.set_title(f"{st.session_state.ticker} Daily — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="Price", linewidth=1.4)
            ax.plot(ema30_show.index, ema30_show.values, "--", alpha=0.4, linewidth=1.0, label="_nolegend_")

            if show_bbands and not bb_up_d2_show.dropna().empty and not bb_lo_d2_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d2_show, bb_up_d2_show, alpha=0.04, label="_nolegend_")
                ax.plot(bb_mid_d2_show.index, bb_mid_d2_show.values, "-", linewidth=0.9, alpha=0.35, label="_nolegend_")

            if show_ichi and not kijun_d2_show.dropna().empty:
                ax.plot(kijun_d2_show.index, kijun_d2_show.values, "-", linewidth=1.2, color="black",
                        alpha=0.55, label="Kijun")

            if show_hma and not hma_d2_full.dropna().empty:
                ax.plot(hma_d2_full.index, hma_d2_full.values, "-", linewidth=1.3, alpha=0.9, label="HMA")

            if show_psar and not psar_d2_show.dropna().empty:
                ax.plot(psar_d2_show.index, psar_d2_show.values, "-", linewidth=1.8, color="purple", alpha=0.95, label="PSAR", zorder=6)

            if not yhat_d_show.empty:
                slope_col_d2 = "tab:green" if m_d >= 0 else "tab:red"
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=3.2, color=slope_col_d2, label="Trend")
            if not up_d_show.empty and not lo_d_show.empty:
                ax.plot(up_d_show.index, up_d_show.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")
            if not lo_d_show.empty:
                ax.plot(lo_d_show.index, lo_d_show.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="")

            badges_top2 = []
            band_sig_d2 = last_band_reversal_signal(price=df_show, band_upper=up_d_show, band_lower=lo_d_show,
                                                    trend_slope=m_d, prox=sr_prox_pct, confirm_bars=rev_bars_confirm)
            # KEEP BUY Band REV only. REMOVE SELL Band REV callout.
            if band_sig_d2 is not None and band_sig_d2.get("side") == "BUY":
                badges_top2.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_d2['price'])}", "tab:green"))
                annotate_buy_triangle(ax, band_sig_d2["time"], band_sig_d2["price"])

            try:
                res_val_d2 = float(res30_show.iloc[-1])
                sup_val_d2 = float(sup30_show.iloc[-1])
                if np.isfinite(res_val_d2) and np.isfinite(sup_val_d2):
                    ax.hlines(res_val_d2, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:red", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                    ax.hlines(sup_val_d2, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:green", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                    label_on_left(ax, res_val_d2, f"R {fmt_price_val(res_val_d2)}", color="tab:red")
                    label_on_left(ax, sup_val_d2, f"S {fmt_price_val(sup_val_d2)}", color="tab:green")
            except Exception:
                res_val_d2 = sup_val_d2 = np.nan

            star_d2 = last_reversal_star(df_show, trend_slope=m_d, lookback=20, confirm_bars=rev_bars_confirm)
            if star_d2 is not None:
                annotate_star(ax, star_d2["time"], star_d2["price"], star_d2["kind"], show_text=False)
                if star_d2.get("kind") == "trough":
                    badges_top2.append((f"★ Trough REV @{fmt_price_val(star_d2['price'])}", "tab:green"))
                elif star_d2.get("kind") == "peak":
                    badges_top2.append((f"★ Peak REV @{fmt_price_val(star_d2['price'])}", "tab:red"))

            hma_cross_star2 = last_hma_cross_star(df_show, hma_d2_full, trend_slope=m_d, lookback=30)
            if hma_cross_star2 is not None:
                if hma_cross_star2["kind"] == "trough":
                    annotate_star(ax, hma_cross_star2["time"], hma_cross_star2["price"], hma_cross_star2["kind"], show_text=False, color_override="black")
                    badges_top2.append((f"★ Buy HMA Cross @{fmt_price_val(hma_cross_star2['price'])}", "black"))
                else:
                    annotate_star(ax, hma_cross_star2["time"], hma_cross_star2["price"], hma_cross_star2["kind"], show_text=False, color_override="tab:blue")
                    badges_top2.append((f"★ Sell HMA Cross @{fmt_price_val(hma_cross_star2['price'])}", "tab:blue"))

            breakout_d2 = last_breakout_signal(
                price=df_show, resistance=res30_show, support=sup30_show,
                prox=sr_prox_pct, confirm_bars=rev_bars_confirm
            )
            if breakout_d2 is not None:
                if breakout_d2["dir"] == "UP":
                    annotate_breakout(ax, breakout_d2["time"], breakout_d2["price"], "UP")
                    badges_top2.append((f"▲ BREAKOUT @{fmt_price_val(breakout_d2['price'])}", "tab:green"))
                else:
                    annotate_breakout(ax, breakout_d2["time"], breakout_d2["price"], "DOWN")
                    badges_top2.append((f"▼ BREAKDOWN @{fmt_price_val(breakout_d2['price'])}", "tab:red"))

            sr99_sig2 = daily_sr_99_reversal_signal(
                price=df_show,
                support=sup30_show,
                resistance=res30_show,
                upper99=up_d_show,
                lower99=lo_d_show,
                trend_slope=m_d,
                prox=sr_prox_pct,
                confirm_bars=rev_bars_confirm
            )
            if sr99_sig2 is not None:
                if sr99_sig2["side"] == "BUY":
                    annotate_signal_box(ax, sr99_sig2["time"], sr99_sig2["price"], side="BUY", note=sr99_sig2["note"])
                    badges_top2.append((f"▲ BUY ALERT 99% SR REV @{fmt_price_val(sr99_sig2['price'])}", "tab:green"))
                else:
                    annotate_signal_box(ax, sr99_sig2["time"], sr99_sig2["price"], side="SELL", note=sr99_sig2["note"])
                    badges_top2.append((f"▼ SELL ALERT 99% SR REV @{fmt_price_val(sr99_sig2['price'])}", "tab:red"))

            draw_top_badges(ax, badges_top2)

            # --- NEW (Addition only): MACD normalized-to-price overlay (Tab 2 Daily) ---
            if show_macd:
                try:
                    macd_line_d2, macd_sig_d2, _ = compute_macd(df_show, fast=macd_fast, slow=macd_slow, signal=macd_signal)
                    macd_ratio_d2 = (macd_line_d2 / _coerce_1d_series(df_show).replace(0, np.nan))
                    max_abs_d2 = float(np.nanmax(np.abs(macd_ratio_d2.to_numpy(dtype=float)))) if len(macd_ratio_d2) else np.nan
                    if np.isfinite(max_abs_d2) and max_abs_d2 > 0:
                        ymin, ymax = ax.get_ylim()
                        pr = (ymax - ymin) if (np.isfinite(ymax) and np.isfinite(ymin)) else 1.0
                        base = ymin + pr * 0.12
                        amp  = pr * 0.10
                        macd_overlay_d2 = base + (macd_ratio_d2 / max_abs_d2) * amp
                        ax.plot(df_show.index, macd_overlay_d2.values, "-", linewidth=1.05, alpha=0.75, label="MACD (norm)")
                        ax.hlines(base, xmin=df_show.index[0], xmax=df_show.index[-1],
                                  colors="grey", linestyles="--", linewidth=0.7, alpha=0.25, label="_nolegend_")
                except Exception:
                    pass

            # TOP instruction banner (Daily) — LOCAL=m_d, GLOBAL=m_d
            try:
                px_val_d2  = float(df_show.iloc[-1])
                confirm_side2 = sr99_sig2["side"] if sr99_sig2 is not None else None
                draw_instruction_ribbons(ax, m_d, sup_val_d2, res_val_d2, px_val_d2, st.session_state.ticker,
                                         confirm_side=confirm_side2,
                                         global_slope=m_d)
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

        if view in ("Intraday","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                st.info("Intraday view is rendered fully in Tab 1 (same logic).")

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = fetch_close_df_period(st.session_state.ticker, bb_period)
        if df3.empty or 'Close' not in df3:
            st.warning("Not enough historical data to compute Bull vs Bear summary.")
        else:
            df3['PctChange'] = df3['Close'].pct_change()
            df3['Bull'] = df3['PctChange'] > 0
            bull = int(df3['Bull'].sum())
            bear = int((~df3['Bull']).sum())
            total = bull + bear if (bull + bear) > 0 else 1
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Days", bull + bear)
            c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
            c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
            c4.metric("Lookback", bb_period)

# --- Tab 4: Metrics ---
with tab4:
    st.header("Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df_hist = fetch_hist(st.session_state.ticker)
        last_price = _safe_last_float(df_hist)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.subheader(f"Last 3 Months  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        trend3m, up3m, lo3m, m3m, r2_3m = regression_with_band(df3m, lookback=len(df3m))

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", linewidth=2.0, color="tab:red", alpha=0.9, label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", linewidth=2.0, color="tab:green", alpha=0.9, label="Support")
        if not trend3m.empty:
            col3 = "tab:green" if m3m >= 0 else "tab:red"
            ax.plot(trend3m.index, trend3m.values, "--", color=col3, linewidth=3.0,
                    label=f"Trend (m={fmt_slope(m3m)}/bar)")
        if not up3m.empty and not lo3m.empty:
            ax.plot(up3m.index, up3m.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend +2σ")
            ax.plot(lo3m.index, lo3m.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend -2σ")
        ax.set_xlabel("Date (PST)")
        ax.text(0.99, 0.02,
                f"R² (3M): {fmt_r2(r2_3m)}  •  Slope: {fmt_slope(m3m)}/bar",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax.legend()
        pad_right_xaxis(ax, frac=0.06)
        st.pyplot(fig)

        st.markdown("---")
        df0 = fetch_close_df_period(st.session_state.ticker, bb_period)
        if df0.empty or 'Close' not in df0:
            st.warning("Not enough data to compute metrics for the selected lookback.")
        else:
            df0['PctChange'] = df0['Close'].pct_change()
            df0['Bull'] = df0['PctChange'] > 0
            df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

            st.subheader("Close + 30-day MA + Trend")
            res0 = df0['Close'].rolling(30, min_periods=1).max()
            sup0 = df0['Close'].rolling(30, min_periods=1).min()
            trend0, up0, lo0, m0, r2_0 = regression_with_band(df0['Close'], lookback=len(df0))

            fig0, ax0 = plt.subplots(figsize=(14,5))
            ax0.plot(df0.index, df0['Close'], label="Close")
            ax0.plot(df0.index, df0['MA30'], label="30 MA")
            ax0.plot(res0.index, res0, ":", linewidth=2.0, color="tab:red", alpha=0.9, label="Resistance")
            ax0.plot(sup0.index, sup0, ":", linewidth=2.0, color="tab:green", alpha=0.9, label="Support")
            if not trend0.empty:
                col0 = "tab:green" if m0 >= 0 else "tab:red"
                ax0.plot(trend0.index, trend0.values, "--", color=col0, linewidth=3.0,
                         label=f"Trend (m={fmt_slope(m0)}/bar)")
            if not up0.empty and not lo0.empty:
                ax0.plot(up0.index, up0.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend +2σ")
                ax0.plot(lo0.index, lo0.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend -2σ")
            ax0.set_xlabel("Date (PST)")
            ax0.text(0.99, 0.02,
                     f"R² ({bb_period}): {fmt_r2(r2_0)}  •  Slope: {fmt_slope(m0)}/bar",
                     transform=ax0.transAxes,
                     ha="right", va="bottom",
                     fontsize=9, color="black",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax0.legend()
            pad_right_xaxis(ax0, frac=0.06)
            st.pyplot(fig0)

            st.markdown("---")
            st.subheader("Daily % Change")
            st.line_chart(df0['PctChange'], use_container_width=True)

            st.subheader("Bull/Bear Distribution")
            dist = pd.DataFrame({
                "Type": ["Bull", "Bear"],
                "Days": [int(df0['Bull'].sum()),
                         int((~df0['Bull']).sum())]
            }).set_index("Type")
            st.bar_chart(dist, use_container_width=True)

# --- Tab 5: NTD -0.75 Scanner (Latest NTD < -0.75) ---
with tab5:
    st.header("NTD -0.75 Scanner (NTD < -0.75)")
    st.caption("Scans the universe for symbols whose **latest NTD value** is below **-0.75** "
               "on the Daily NTD line (and on the Hourly NTD line for Forex).")

    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox("Hourly lookback for Forex:",
                                   ["24h", "48h", "96h"],
                                   index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
                                   key="ntd_scan_hour_range")
    scan_period = period_map[scan_hour_range]
    thresh = -0.75
    run = st.button("Scan Universe", key="btn_ntd_scan")

    def compute_normalized_trend(close: pd.Series, window: int = 60) -> pd.Series:
        s = _coerce_1d_series(close).astype(float)
        if s.empty or window < 3:
            return pd.Series(index=s.index, dtype=float)
        minp = max(5, window // 3)
        def _slope(y: pd.Series) -> float:
            y = pd.Series(y).dropna()
            if len(y) < 3: return np.nan
            x = np.arange(len(y), dtype=float)
            try: m, _ = np.polyfit(x, y.to_numpy(dtype=float), 1)
            except Exception: return np.nan
            return float(m)
        slope_roll = s.rolling(window, min_periods=minp).apply(_slope, raw=False)
        vol = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
        ntd_raw = (slope_roll * window) / vol
        return np.tanh(ntd_raw / 2.0).reindex(s.index)

    if run:
        daily_rows = []
        for sym in universe:
            try:
                s = fetch_hist(sym)
                ntd = compute_normalized_trend(s, window=ntd_window).dropna()
                ntd_val = float(ntd.iloc[-1]) if not ntd.empty else np.nan
                ts = ntd.index[-1] if not ntd.empty else None
                close_val = _safe_last_float(s)
            except Exception:
                ntd_val, ts, close_val = np.nan, None, np.nan
            daily_rows.append({"Symbol": sym, "NTD_Last": ntd_val,
                               "BelowThresh": (np.isfinite(ntd_val) and ntd_val < thresh),
                               "Close": close_val, "Timestamp": ts})
        df_daily = pd.DataFrame(daily_rows)
        hits_daily = df_daily[df_daily["BelowThresh"] == True].copy().sort_values("NTD_Last")

        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric(f"Daily NTD < {thresh:+.2f}", int(hits_daily.shape[0]))

        st.subheader(f"Daily — latest NTD < {thresh:+.2f}")
        if hits_daily.empty:
            st.info(f"No symbols where the latest **daily** NTD value is below {thresh:+.2f}.")
        else:
            view = hits_daily.copy()
            view["NTD_Last"] = view["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
            view["Close"] = view["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view[["Symbol","Timestamp","Close","NTD_Last"]].reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.subheader(f"Daily — Kijun Up-Cross + Upward Slope (latest bar, Kijun={ichi_base})")
        kij_rows = []
        for sym in universe:
            try:
                ohlc = fetch_hist_ohlc(sym)
                if ohlc is None or ohlc.empty or not {'High','Low','Close'}.issubset(ohlc.columns):
                    continue
                _, _, _, m_sym, _ = regression_with_band(ohlc["Close"], slope_lb_daily)
                if not np.isfinite(m_sym) or m_sym <= 0:
                    continue
                _, kijun, _, _, _ = ichimoku_lines(ohlc["High"], ohlc["Low"], ohlc["Close"],
                                                   conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun = kijun.ffill().bfill().reindex(ohlc.index)
                close = ohlc["Close"].astype(float).reindex(ohlc.index)
                mask = close.notna() & kijun.notna()
                if mask.sum() < 2:
                    continue
                c_prev, c_now = float(close[mask].iloc[-2]), float(close[mask].iloc[-1])
                k_prev, k_now = float(kijun[mask].iloc[-2]), float(kijun[mask].iloc[-1])
                up_cross = (c_prev < k_prev) and (c_now >= k_now)
                if up_cross:
                    kij_rows.append({
                        "Symbol": sym,
                        "Timestamp": close[mask].index[-1],
                        "Close": c_now,
                        "Kijun": k_now,
                        "Slope": m_sym
                    })
            except Exception:
                pass

        if not kij_rows:
            st.info("No daily symbols just crossed **up through the Kijun** while in an **upward slope** on the latest bar.")
        else:
            df_kij = pd.DataFrame(kij_rows).sort_values("Symbol")
            show_kij = df_kij.copy()
            show_kij["Close"] = show_kij["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            show_kij["Kijun"] = show_kij["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            show_kij["Slope"] = show_kij["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            st.dataframe(show_kij[["Symbol","Timestamp","Close","Kijun","Slope"]].reset_index(drop=True),
                         use_container_width=True)

        if mode == "Forex":
            st.markdown("---")
            st.subheader(f"Forex Hourly — latest NTD < {thresh:+.2f} ({scan_hour_range} lookback)")
            hourly_rows = []
            for sym in universe:
                try:
                    intr = fetch_intraday(sym, period=scan_period)
                    close_val_h = _safe_last_float(intr["Close"]) if intr is not None and "Close" in intr else np.nan
                    ntd = compute_normalized_trend(intr["Close"], window=ntd_window) if (intr is not None and "Close" in intr) else pd.Series(dtype=float)
                    ntd_val_h = float(ntd.dropna().iloc[-1]) if not ntd.dropna().empty else np.nan
                    ts_h = ntd.dropna().index[-1] if not ntd.dropna().empty else None
                except Exception:
                    close_val_h, ntd_val_h, ts_h = np.nan, np.nan, None
                hourly_rows.append({"Symbol": sym, "NTD_Last": ntd_val_h, "BelowThresh": (np.isfinite(ntd_val_h) and ntd_val_h < thresh),
                                    "Close": close_val_h, "Timestamp": ts_h})
            df_hour = pd.DataFrame(hourly_rows)
            hits_hour = df_hour[df_hour["BelowThresh"] == True].copy().sort_values("NTD_Last")

            if hits_hour.empty:
                st.info(f"No Forex pairs where the latest **hourly** NTD value is below {thresh:+.2f} within {scan_hour_range} lookback.")
            else:
                showh = hits_hour.copy()
                showh["NTD_Last"] = showh["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
                showh["Close"] = showh["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(showh[["Symbol","Timestamp","Close","NTD_Last"]].reset_index(drop=True), use_container_width=True)

            st.subheader(f"Forex Hourly — Price > Ichimoku Kijun({ichi_base}) (latest bar, {scan_hour_range})")
            habove_rows = []
            for sym in universe:
                try:
                    intr = fetch_intraday(sym, period=scan_period)
                    if intr is None or intr.empty or not {'High','Low','Close'}.issubset(intr.columns):
                        above_h, ts_h, close_h, kij_h = False, None, np.nan, np.nan
                    else:
                        _, kij, _, _, _ = ichimoku_lines(intr["High"], intr["Low"], intr["Close"], base=ichi_base)
                        kij = kij.ffill().bfill().reindex(intr.index)
                        close = intr["Close"].astype(float).reindex(intr.index)
                        mask = close.notna() & kij.notna()
                        if mask.sum() < 1:
                            above_h, ts_h, close_h, kij_h = False, None, np.nan, np.nan
                        else:
                            close_h = float(close[mask].iloc[-1]); kij_h = float(kij[mask].iloc[-1])
                            ts_h = close[mask].index[-1]; above_h = (close_h > kij_h)
                except Exception:
                    above_h, ts_h, close_h, kij_h = False, None, np.nan, np.nan
                habove_rows.append({"Symbol": sym, "AboveNow": above_h, "Timestamp": ts_h, "Close": close_h, "Kijun": kij_h})
            df_above_hour = pd.DataFrame(habove_rows)
            df_above_hour = df_above_hour[df_above_hour["AboveNow"] == True]
            if df_above_hour.empty:
                st.info("No Forex pairs with Price > Kijun on the latest bar.")
            else:
                vch = df_above_hour.copy()
                vch["Close"] = vch["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                vch["Kijun"] = vch["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(vch[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

# --- Tab 6: Long-Term History ---
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")
    default_idx = 0
    if st.session_state.get("ticker") in universe:
        default_idx = universe.index(st.session_state["ticker"])
    sym = st.selectbox("Ticker:", universe, index=default_idx, key="hist_long_ticker")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("5Y", key="btn_5y"):  st.session_state.hist_years = 5
    if c2.button("10Y", key="btn_10y"): st.session_state.hist_years = 10
    if c3.button("15Y", key="btn_15y"): st.session_state.hist_years = 15
    if c4.button("20Y", key="btn_20y"): st.session_state.hist_years = 20

    years = int(st.session_state.hist_years)
    st.caption(f"Showing last **{years} years**. Support/Resistance = rolling **252-day** extremes; trendline fits the shown window.")

    s_full = fetch_hist_max(sym)
    if s_full is None or s_full.empty:
        st.warning("No historical data available.")
    else:
        end_ts = s_full.index.max()
        start_ts = end_ts - pd.DateOffset(years=years)
        s = s_full[s_full.index >= start_ts]
        if s.empty:
            st.warning(f"No data in the last {years} years for {sym}.")
        else:
            res_roll = s.rolling(252, min_periods=1).max()
            sup_roll = s.rolling(252, min_periods=1).min()
            res_last = float(res_roll.iloc[-1]) if len(res_roll) else np.nan
            sup_last = float(sup_roll.iloc[-1]) if len(sup_roll) else np.nan
            yhat_all, upper_all, lower_all, m_all, r2_all = regression_with_band(s, lookback=len(s))

            fig, ax = plt.subplots(figsize=(14,5))
            plt.subplots_adjust(right=0.995, left=0.06, top=0.92)
            ax.set_title(f"{sym} — Last {years} Years — Price + 252d S/R + Trend")
            ax.plot(s.index, s.values, label="Close", linewidth=1.4)
            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:red",   linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:green", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")
            if not yhat_all.empty:
                col_all = "tab:green" if m_all >= 0 else "tab:red"
                ax.plot(yhat_all.index, yhat_all.values, "--",
                        linewidth=3.2, color=col_all, label="Trend")
            if not upper_all.empty and not lower_all.empty:
                ax.plot(upper_all.index, upper_all.values, ":", linewidth=3.0,
                        color="black", alpha=1.0, label="_nolegend_")
                ax.plot(lower_all.index, lower_all.values, ":", linewidth=3.0,
                        color="black", alpha=1.0, label="_nolegend_")

            px_now = _safe_last_float(s)
            price_line = f"Current price: {fmt_price_val(px_now)}" if np.isfinite(px_now) else ""
            footer = (price_line + ("\n" if price_line else "") +
                      f"R² (trend): {fmt_r2(r2_all)}  •  Slope: {fmt_slope(m_all)}/bar")
            ax.text(0.99, 0.02, footer,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="grey", alpha=0.7))

            _simplify_axes(ax)
            ax.set_xlabel("Date (PST)")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.4)
            pad_right_xaxis(ax, frac=0.06)
            st.pyplot(fig)

# --- Tab 7: Upward Slope Stickers (UPDATED) ---
with tab7:
    st.header("Upward Slope Stickers")
    st.caption("Lists symbols whose **Daily** regression slope is **upward** (m>0) and the **latest close** is **below the trendline** (price < fitted slope).")

    run99 = st.button("Scan Universe for Upward Slope & Below-Slope", key="btn_scan_upslope_below")

    if run99:
        rows = []
        for sym in universe:
            try:
                s = fetch_hist(sym)
                if s is None or s.dropna().shape[0] < max(3, slope_lb_daily):
                    continue
                yhat_s, up99, lo99, m_sym, r2_sym = regression_with_band(s, lookback=slope_lb_daily, z=Z_FOR_99)
                last_close = float(s.dropna().iloc[-1]) if s.dropna().shape[0] else np.nan
                last_time  = s.dropna().index[-1] if s.dropna().shape[0] else None
                yhat_last  = float(yhat_s.iloc[-1]) if not yhat_s.empty else np.nan
                if np.isfinite(m_sym) and m_sym > 0 and np.isfinite(last_close) and np.isfinite(yhat_last) and (last_close < yhat_last):
                    gap = yhat_last - last_close
                    gap_pct = gap / yhat_last if yhat_last != 0 else np.nan
                    rows.append({
                        "Symbol": sym,
                        "Timestamp": last_time,
                        "Close": last_close,
                        "Trendline": yhat_last,
                        "Gap": gap,
                        "GapPct": gap_pct,
                        "Slope": m_sym,
                        "R2": r2_sym
                    })
            except Exception:
                pass

        if not rows:
            st.info("No symbols currently have an **upward daily slope** with **price below the slope** on the latest bar.")
        else:
            out = pd.DataFrame(rows).sort_values(["GapPct","Symbol"], ascending=[False, True])
            view = out.copy()
            view["Close"] = view["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["Trendline"] = view["Trendline"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["Gap"] = view["Gap"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["GapPct"] = view["GapPct"].map(lambda v: fmt_pct(v, 2) if np.isfinite(v) else "n/a")
            view["Slope"] = view["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view["R2"] = view["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            st.dataframe(view[["Symbol","Timestamp","Close","Trendline","Gap","GapPct","Slope","R2"]]
                         .reset_index(drop=True),
                         use_container_width=True)

# --- Tab 8: Daily Support Reversals (NEW) ---
with tab8:
    st.header("Daily Support Reversals")
    st.caption(
        "Scans for symbols that **touched daily Support** (rolling 30-bar **Close** min) within your "
        "**S/R / Band proximity (%)** and then printed **consecutive higher closes** (uses your "
        "**Consecutive bars to confirm** setting)."
    )

    if st.button("Scan Universe for Daily Support Reversals", key="btn_scan_support_rev"):
        rows = []
        for sym in universe:
            try:
                s = fetch_hist(sym)
                if s is None or s.dropna().shape[0] < max(10, slope_lb_daily):
                    continue
                sup30 = s.rolling(30, min_periods=1).min()
                sig = find_support_touch_confirmed_up(
                    price=s,
                    support=sup30,
                    prox=sr_prox_pct,
                    confirm_bars=rev_bars_confirm,
                    lookback_bars=30
                )
                if sig is None:
                    continue
                _, _, _, m_sym, r2_sym = regression_with_band(s, lookback=slope_lb_daily, z=Z_FOR_99)
                rows.append({
                    "Symbol": sym,
                    "Touched": sig["t_touch"],
                    "Close@Touch": sig["touch_close"],
                    "Support@Touch": sig["support"],
                    "Now": s.dropna().index[-1],
                    "NowClose": sig["now_close"],
                    "RisePct": sig["gain_pct"],
                    "BarsSince": sig["bars_since_touch"],
                    "Slope": m_sym,
                    "R2": r2_sym
                })
            except Exception:
                pass

        if not rows:
            st.info("No symbols met the **support-touch → confirmed up** criteria at this time.")
        else:
            df = pd.DataFrame(rows).sort_values(["RisePct","BarsSince"], ascending=[False, True])

            view = df.copy()
            view["Close@Touch"] = view["Close@Touch"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["Support@Touch"] = view["Support@Touch"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["NowClose"] = view["NowClose"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["RisePct"] = view["RisePct"].map(lambda v: fmt_pct(v, 2) if np.isfinite(v) else "n/a")
            view["Slope"] = view["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view["R2"] = view["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")

            st.dataframe(
                view[["Symbol","Touched","Close@Touch","Support@Touch","Now","NowClose","RisePct","BarsSince","Slope","R2"]]
                .reset_index(drop=True),
                use_container_width=True
            )
