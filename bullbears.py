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

            # Trendline with ±2σ band and R² (Daily)
            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, _, _ = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            df_show     = subset_by_daily_view(df, daily_view)
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

            # HMA lines (no HMA signal callouts)
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

            if not yhat_d_show.empty:
                slope_col_d = "tab:green" if m_d >= 0 else "tab:red"
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2.0, color=slope_col_d, label="Trend")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, ":", linewidth=1.8, color="black", alpha=0.6, label="_nolegend_")
                ax.plot(lower_d_show.index, lower_d_show.values, ":", linewidth=1.8, color="black", alpha=0.6, label="_nolegend_")
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="")

            # DAILY — Support/Resistance lines
            try:
                res_val_d = float(res30_show.iloc[-1])
                sup_val_d = float(sup30_show.iloc[-1])
                if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                    ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:red", linestyles="-", linewidth=1.3, alpha=0.65, label="_nolegend_")
                    ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:green", linestyles="-", linewidth=1.3, alpha=0.65, label="_nolegend_")
                    label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
                    label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")
            except Exception:
                res_val_d = sup_val_d = np.nan

            # --- Signals to badges: BUY Band REV & Star REVs; plus star/triangle rendering ---
            badges_top = []

            # Band REV (Daily)
            band_sig_d = last_band_reversal_signal(
                price=df_show, band_upper=upper_d_show, band_lower=lower_d_show,
                trend_slope=m_d, prox=sr_prox_pct, confirm_bars=rev_bars_confirm
            )
            if band_sig_d is not None and band_sig_d.get("side") == "BUY":
                badges_top.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_d['price'])}", "tab:green"))
                annotate_buy_triangle(ax, band_sig_d["time"], band_sig_d["price"])
            elif band_sig_d is not None and band_sig_d.get("side") == "SELL":
                annotate_band_rev_outside(ax, band_sig_d["time"], band_sig_d["price"], band_sig_d["side"], note=band_sig_d.get("note",""))

            # Star (Daily) — draw on chart for BOTH kinds; trough/peak also get top badges
            star_d = last_reversal_star(df_show, trend_slope=m_d, lookback=20, confirm_bars=rev_bars_confirm)
            if star_d is not None:
                annotate_star(ax, star_d["time"], star_d["price"], star_d["kind"], show_text=(star_d["kind"] == "peak"))
                if star_d.get("kind") == "trough":
                    badges_top.append((f"★ Trough REV @{fmt_price_val(star_d['price'])}", "tab:green"))
                elif star_d.get("kind") == "peak":
                    badges_top.append((f"★ Peak REV @{fmt_price_val(star_d['price'])}", "tab:red"))

            # Draw compact badges
            draw_top_badges(ax, badges_top)

            # TOP instruction banner (Daily)
            try:
                px_val_d  = float(df_show.iloc[-1])
                draw_instruction_ribbons(ax, m_d, sup_val_d, res_val_d, px_val_d, sel)
            except Exception:
                pass

            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            _simplify_axes(ax)
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.4)
            st.pyplot(fig)

        # ----- Hourly (Price only) -----
        if chart in ("Hourly","Both"):
            intraday = st.session_state.intraday
            if intraday is None or intraday.empty or "Close" not in intraday:
                st.warning("No intraday data available.")
            else:
                hc = intraday["Close"].astype(float).ffill()
                # Robust linear fit
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
                psar_h_df = psar_h_df.reindex(hc.index)

                # Hourly regression slope & bands (for signals)
                yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
                slope_sig_h = m_h if np.isfinite(m_h) else slope_h

                # GLOBAL (DAILY) slope for instruction order
                try:
                    df_global = st.session_state.df_hist
                    _, _, _, m_global, _ = regression_with_band(df_global, slope_lb_daily)
                except Exception:
                    m_global = slope_sig_h

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.86, right=0.995, left=0.06)

                trend_color = "tab:green" if slope_h >= 0 else "tab:red"
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")

                ax2.plot(hc.index, hc, label="Price", linewidth=1.2)
                ax2.plot(hc.index, he, "--", alpha=0.45, linewidth=0.9, label="_nolegend_")
                ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=1.6, color=trend_color, alpha=0.75)

                if show_hma and not hma_h.dropna().empty:
                    ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.3, alpha=0.9, label="HMA")

                if show_ichi and not kijun_h.dropna().empty:
                    ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.1, color="black", alpha=0.55, label="Kijun")

                if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
                    ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.04, label="_nolegend_")
                    ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=0.8, alpha=0.3, label="_nolegend_")

                res_val = sup_val = px_val = np.nan
                try:
                    res_val = float(res_h.iloc[-1]); sup_val = float(sup_h.iloc[-1]); px_val = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red",   linestyles="-", linewidth=1.2, alpha=0.6, label="_nolegend_")
                    ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.2, alpha=0.6, label="_nolegend_")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                # --- Signals to badges + stars/triangles (Hourly) ---
                badges_top_h = []

                band_sig_h = last_band_reversal_signal(
                    price=hc, band_upper=upper_h, band_lower=lower_h,
                    trend_slope=m_h, prox=sr_prox_pct, confirm_bars=rev_bars_confirm
                )
                if band_sig_h is not None and band_sig_h.get("side") == "BUY":
                    badges_top_h.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_h['price'])}", "tab:green"))
                    annotate_buy_triangle(ax2, band_sig_h["time"], band_sig_h["price"])
                elif band_sig_h is not None and band_sig_h.get("side") == "SELL":
                    annotate_band_rev_outside(ax2, band_sig_h["time"], band_sig_h["price"], band_sig_h["side"], note=band_sig_h.get("note",""))

                star_h = last_reversal_star(hc, trend_slope=m_h, lookback=20, confirm_bars=rev_bars_confirm)
                if star_h is not None:
                    annotate_star(ax2, star_h["time"], star_h["price"], star_h["kind"], show_text=(star_h["kind"] == "peak"))
                    if star_h.get("kind") == "trough":
                        badges_top_h.append((f"★ Trough REV @{fmt_price_val(star_h['price'])}", "tab:green"))
                    elif star_h.get("kind") == "peak":
                        badges_top_h.append((f"★ Peak REV @{fmt_price_val(star_h['price'])}", "tab:red"))

                draw_top_badges(ax2, badges_top_h)

                # TOP instruction banner (Hourly) — use GLOBAL daily slope for order
                draw_instruction_ribbons(ax2, m_global, sup_val, res_val, px_val, sel)

                # footer stats
                if np.isfinite(px_val):
                    nbb_txt = ""
                    try:
                        last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        if np.isfinite(last_nbb) and np.isfinite(last_pct):
                            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                    except Exception:
                        pass
                    ax2.text(0.99, 0.02, f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                             transform=ax2.transAxes, ha="right", va="bottom",
                             fontsize=10, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if not st_line_intr.dropna().empty:
                    ax2.plot(st_line_intr.index, st_line_intr.values, "-", alpha=0.6, label="_nolegend_")
                if not yhat_h.empty:
                    slope_col_h = "tab:green" if m_h >= 0 else "tab:red"
                    ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=1.8, color=slope_col_h, alpha=0.8, label="Slope Fit")
                if not upper_h.empty and not lower_h.empty:
                    ax2.plot(upper_h.index, upper_h.values, ":", linewidth=1.5, color="black", alpha=0.5, label="_nolegend_")
                    ax2.plot(lower_h.index, lower_h.values, ":", linewidth=1.5, color="black", alpha=0.5, label="_nolegend_")

                if mode == "Forex" and show_sessions_pst and not hc.empty:
                    sess = compute_session_lines(hc.index)
                    draw_session_lines(ax2, sess)

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=0.9, alpha=0.35)
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" {lbl}", va="center", fontsize=8, alpha=0.6)

                _simplify_axes(ax2)
                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.4)
                st.pyplot(fig2)

        # News table
        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        # Forecast table
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
            yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)

            kijun_d2 = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d2, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                      conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d2 = kijun_d2.ffill().bfill()

            bb_mid_d2, bb_up_d2, bb_lo_d2 = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)[:3]

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

            # HMA plotting (signals removed)
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

            if not yhat_d_show.empty:
                slope_col_d2 = "tab:green" if m_d >= 0 else "tab:red"
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2.0, color=slope_col_d2, label="Trend")
            if not up_d_show.empty and not lo_d_show.empty:
                ax.plot(up_d_show.index, up_d_show.values, ":", linewidth=1.8, color="black", alpha=0.6, label="_nolegend_")
            if not lo_d_show.empty:
                ax.plot(lo_d_show.index, lo_d_show.values, ":", linewidth=1.8, color="black", alpha=0.6, label="_nolegend_")
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="")

            # Band signal (Daily) — top badges + stars/triangle
            badges_top2 = []
            band_sig_d2 = last_band_reversal_signal(price=df_show, band_upper=up_d_show, band_lower=lo_d_show,
                                                    trend_slope=m_d, prox=sr_prox_pct, confirm_bars=rev_bars_confirm)
            if band_sig_d2 is not None and band_sig_d2.get("side") == "BUY":
                badges_top2.append((f"▲ BUY Band REV @{fmt_price_val(band_sig_d2['price'])}", "tab:green"))
                annotate_buy_triangle(ax, band_sig_d2["time"], band_sig_d2["price"])
            elif band_sig_d2 is not None and band_sig_d2.get("side") == "SELL":
                annotate_band_rev_outside(ax, band_sig_d2["time"], band_sig_d2["price"], band_sig_d2["side"], note=band_sig_d2.get("note",""))

            # DAILY — Support/Resistance horizontal lines
            try:
                res_val_d2 = float(res30_show.iloc[-1])
                sup_val_d2 = float(sup30_show.iloc[-1])
                if np.isfinite(res_val_d2) and np.isfinite(sup_val_d2):
                    ax.hlines(res_val_d2, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:red", linestyles="-", linewidth=1.3, alpha=0.65, label="_nolegend_")
                    ax.hlines(sup_val_d2, xmin=df_show.index[0], xmax=df_show.index[-1],
                              colors="tab:green", linestyles="-", linewidth=1.3, alpha=0.65, label="_nolegend_")
                    label_on_left(ax, res_val_d2, f"R {fmt_price_val(res_val_d2)}", color="tab:red")
                    label_on_left(ax, sup_val_d2, f"S {fmt_price_val(sup_val_d2)}", color="tab:green")
            except Exception:
                res_val_d2 = sup_val_d2 = np.nan

            # Star (Daily)
            star_d2 = last_reversal_star(df_show, trend_slope=m_d, lookback=20, confirm_bars=rev_bars_confirm)
            if star_d2 is not None:
                annotate_star(ax, star_d2["time"], star_d2["price"], star_d2["kind"], show_text=(star_d2["kind"] == "peak"))
                if star_d2.get("kind") == "trough":
                    badges_top2.append((f"★ Trough REV @{fmt_price_val(star_d2['price'])}", "tab:green"))
                elif star_d2.get("kind") == "peak":
                    badges_top2.append((f"★ Peak REV @{fmt_price_val(star_d2['price'])}", "tab:red"))

            draw_top_badges(ax, badges_top2)

            # TOP instruction banner (Daily)
            try:
                px_val_d2  = float(df_show.iloc[-1])
                draw_instruction_ribbons(ax, m_d, sup_val_d2, res_val_d2, px_val_d2, st.session_state.ticker)
            except Exception:
                pass

            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            _simplify_axes(ax)
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.4)
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
        df3 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull'] = df3['PctChange'] > 0
        bull = int(df3['Bull'].sum())
        bear = int((~df3['Bull']).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
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
        ax.plot(res3m.index, res3m, ":", label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", label="Support")
        if not trend3m.empty:
            col3 = "tab:green" if m3m >= 0 else "tab:red"
            ax.plot(trend3m.index, trend3m.values, "--", color=col3,
                    label=f"Trend (m={fmt_slope(m3m)}/bar)")
        if not up3m.empty and not lo3m.empty:
            ax.plot(up3m.index, up3m.values, ":", linewidth=2.0,
                     color="black", alpha=0.85, label="Trend +2σ")
            ax.plot(lo3m.index, lo3m.values, ":", linewidth=2.0,
                     color="black", alpha=0.85, label="Trend -2σ")
        ax.set_xlabel("Date (PST)")
        ax.text(0.50, 0.02,
                f"R² (3M): {fmt_r2(r2_3m)}",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
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
        ax0.plot(res0.index, res0, ":", label="Resistance")
        ax0.plot(sup0.index, sup0, ":", label="Support")
        if not trend0.empty:
            col0 = "tab:green" if m0 >= 0 else "tab:red"
            ax0.plot(trend0.index, trend0.values, "--", color=col0,
                     label=f"Trend (m={fmt_slope(m0)}/bar)")
        if not up0.empty and not lo0.empty:
            ax0.plot(up0.index, up0.values, ":", linewidth=2.0,
                     color="black", alpha=0.85, label="Trend +2σ")
            ax0.plot(lo0.index, lo0.values, ":", linewidth=2.0,
                     color="black", alpha=0.85, label="Trend -2σ")
        ax0.set_xlabel("Date (PST)")
        ax0.text(0.50, 0.02,
                 f"R² ({bb_period}): {fmt_r2(r2_0)}",
                 transform=ax0.transAxes,
                 ha="center", va="bottom",
                 fontsize=9, color="black",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax0.legend()
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

    # Local NTD (kept only for scanner)
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
        # ---- DAILY: latest NTD < -0.75 ----
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

        # ---- DAILY: PRICE > KIJUN ----
        st.markdown("---")
        st.subheader(f"Daily — Price > Ichimoku Kijun({ichi_base}) (latest bar)")
        def _price_above_kijun_from_df(df: pd.DataFrame, base: int = 26):
            if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
                return False, None, np.nan, np.nan
            ohlc = df[['High','Low','Close']].copy()
            _, kijun, _, _, _ = ichimoku_lines(ohlc['High'], ohlc['Low'], ohlc['Close'], base=base)
            kijun = kijun.ffill().bfill().reindex(ohlc.index)
            close = ohlc['Close'].astype(float).reindex(ohlc.index)
            mask = close.notna() & kijun.notna()
            if mask.sum() < 1:
                return False, None, np.nan, np.nan
            c_now = float(close[mask].iloc[-1]); k_now = float(kijun[mask].iloc[-1])
            ts = close[mask].index[-1]
            above = np.isfinite(c_now) and np.isfinite(k_now) and (c_now > k_now)
            return above, ts if above else None, c_now, k_now

        above_rows = []
        for sym in universe:
            try:
                df_ohlc_sym = fetch_hist_ohlc(sym)
                above, ts, cnow, know = _price_above_kijun_from_df(df_ohlc_sym, base=ichi_base)
            except Exception:
                above, ts, cnow, know = False, None, np.nan, np.nan
            above_rows.append({"Symbol": sym, "AboveNow": above, "Timestamp": ts, "Close": cnow, "Kijun": know})
        df_above_daily = pd.DataFrame(above_rows)
        df_above_daily = df_above_daily[df_above_daily["AboveNow"] == True]
        if df_above_daily.empty:
            st.info("No Daily symbols with Price > Kijun on the latest bar.")
        else:
            v = df_above_daily.copy()
            v["Close"] = v["Close"].map(lambda x: fmt_price_val(x) if np.isfinite(x) else "n/a")
            v["Kijun"] = v["Kijun"].map(lambda x: fmt_price_val(x) if np.isfinite(x) else "n/a")
            st.dataframe(v[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

        # ---- FOREX HOURLY: latest NTD < -0.75 ----
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

            # ---- FOREX HOURLY: PRICE > KIJUN ----
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
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:red",   linestyles="-", linewidth=1.3, alpha=0.6, label="_nolegend_")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:green", linestyles="-", linewidth=1.3, alpha=0.6, label="_nolegend_")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")
            if not yhat_all.empty:
                col_all = "tab:green" if m_all >= 0 else "tab:red"
                ax.plot(yhat_all.index, yhat_all.values, "--",
                        linewidth=2, color=col_all, label="Trend")
            if not upper_all.empty and not lower_all.empty:
                ax.plot(upper_all.index, upper_all.values, ":", linewidth=1.8,
                        color="black", alpha=0.6, label="_nolegend_")
                ax.plot(lower_all.index, lower_all.values, ":", linewidth=1.8,
                        color="black", alpha=0.6, label="_nolegend_")
            px_now = _safe_last_float(s)
            if np.isfinite(px_now):
                ax.text(0.99, 0.02,
                        f"Current price: {fmt_price_val(px_now)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="grey", alpha=0.7))
            ax.text(0.01, 0.02,
                    f"Slope: {fmt_slope(m_all)}/bar",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="grey", alpha=0.7))
            ax.text(0.50, 0.02,
                    f"R² (trend): {fmt_r2(r2_all)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            _simplify_axes(ax)
            ax.set_xlabel("Date (PST)")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.4)
            st.pyplot(fig)
# --- Tab 7: HMA 55 Cross Scanner ---
with tab7:
    st.header("HMA 55 Cross Scanner")
    st.caption(
        "Lists symbols where **price has recently crossed HMA(55)** on the selected timeframe, "
        "with results separated into **Buy / Sell** and each further split by **Upward Trend** and **Downward Trend**."
    )

    # Local scanner controls (tab-only; does not change existing app behavior)
    scan_tf = st.selectbox("Scanner timeframe:", ["Daily", "Hourly (5m)"], index=0, key="hma55_scan_tf")
    recent_bars_hma = st.slider("Recent cross window (bars)", 1, 12, 3, 1, key="hma55_recent_bars")
    scan_hour_range_hma = st.selectbox(
        "Hourly lookback (for Hourly scanner):",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
        key="hma55_scan_hour_range"
    )
    run_hma_scan = st.button("Scan HMA 55 Crosses", key="btn_hma55_scan")

    def _latest_recent_hma_cross(close_like, period: int = 55, recent_bars: int = 3):
        """
        Detect latest recent price/HMA cross within the last `recent_bars` bars.
        Returns dict with:
          {"direction": "UP"|"DOWN", "time", "close", "hma", "bars_ago", "distance"}
        or None.
        """
        close = _coerce_1d_series(close_like).astype(float).dropna()
        if close.empty or close.shape[0] < max(period + 5, 10):
            return None

        hma = compute_hma(close, period=period)
        dfx = pd.DataFrame({"Close": close, "HMA55": hma}).dropna()
        if dfx.shape[0] < 3:
            return None

        diff = dfx["Close"] - dfx["HMA55"]
        prev = diff.shift(1)

        up_cross = (diff > 0) & (prev <= 0)
        dn_cross = (diff < 0) & (prev >= 0)

        cross_events = []
        idx_positions = {ts: i for i, ts in enumerate(dfx.index)}

        for ts in dfx.index[up_cross.fillna(False)]:
            i = idx_positions.get(ts, None)
            if i is None:
                continue
            bars_ago = (len(dfx) - 1) - i
            if bars_ago <= recent_bars - 1:
                cross_events.append({
                    "direction": "UP",   # BUY cross
                    "time": ts,
                    "close": float(dfx.at[ts, "Close"]),
                    "hma": float(dfx.at[ts, "HMA55"]),
                    "bars_ago": int(bars_ago),
                    "distance": float(dfx.at[ts, "Close"] - dfx.at[ts, "HMA55"])
                })

        for ts in dfx.index[dn_cross.fillna(False)]:
            i = idx_positions.get(ts, None)
            if i is None:
                continue
            bars_ago = (len(dfx) - 1) - i
            if bars_ago <= recent_bars - 1:
                cross_events.append({
                    "direction": "DOWN",  # SELL cross
                    "time": ts,
                    "close": float(dfx.at[ts, "Close"]),
                    "hma": float(dfx.at[ts, "HMA55"]),
                    "bars_ago": int(bars_ago),
                    "distance": float(dfx.at[ts, "Close"] - dfx.at[ts, "HMA55"])
                })

        if not cross_events:
            return None
