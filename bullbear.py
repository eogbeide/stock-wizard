# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Lists symbols where the latest NTD is below -0.75 (using latest intraday for hourly; daily uses daily close).")

    scan_frame = st.radio("Frame:", ["Hourly (intraday)", "Daily"], index=0, key=f"ntd_scan_frame_{mode}")
    run_scan = st.button("Run Scanner", key=f"btn_run_ntd_scan_{mode}")

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

# ---------------------------
# TAB 6: LONG-TERM HISTORY
# ---------------------------
with tab6:
    st.header("Long-Term History")
    sel_lt = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
    try:
        smax = fetch_hist_max(sel_lt)
    except Exception:
        smax = pd.Series(dtype=float)

    if smax is None or smax.dropna().empty:
        st.warning("No long-term history available.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{sel_lt} — Max History")
        ax.plot(smax.index, smax.values, label="Close")
        draw_trend_direction_line(ax, smax, label_prefix="Trend (global)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 7: RECENT BUY SCANNER
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner — Daily NPX↑NTD in Uptrend (Stocks + Forex)")
    st.caption(
        "Lists symbols (in the current mode's universe) where **NPX (normalized price)** most recently crossed "
        "**ABOVE** the **NTD** line (the green circle condition) **AND** the DAILY chart-area global trendline "
        "(in the selected Daily view range) is **upward**."
    )

    max_bars = st.slider("Max bars since NPX↑NTD cross", 0, 20, 2, 1, key="buy_scan_npx_max_bars")
    run_buy_scan = st.button("Run Recent BUY Scan", key="btn_run_recent_buy_scan_npx")

    if run_buy_scan:
        rows = []
        for sym in universe:
            r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=ntd_window, daily_view_label=daily_view)
            if r is not None and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)

        if not rows:
            st.info("No recent NPX↑NTD crosses found in an upward daily global trend (within the selected bar window).")
        else:
            out = pd.DataFrame(rows)
            if "Bars Since" in out.columns:
                out["Bars Since"] = out["Bars Since"].astype(int)
            if "Global Slope" in out.columns:
                out["Global Slope"] = out["Global Slope"].astype(float)
            out = out.sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 8: HMA BUY (NEW — replaces removed NPX 0.5-Cross Scanner tab)
# ---------------------------
with tab8:
    st.header("HMA Buy — Daily Price Cross ↑ HMA(55)")
    st.caption(
        "Lists symbols where **price recently crossed ABOVE HMA(55)** on the **Daily** chart within the last **1–3 bars** "
        "(slider). Results are split into:\n"
        "• **(a) Regression > 0**\n"
        "• **(b) Regression < 0**\n\n"
        "Uses the selected **Daily view range** for the regression slope sign."
    )

    c1, c2 = st.columns(2)
    hma_max_bars = c1.slider("Cross must be within last N bars", 1, 3, 2, 1, key="hma_buy_within_n")
    run_hma_buy = c2.button("Run HMA Buy Scan", key=f"btn_run_hma_buy_{mode}")

    if run_hma_buy:
        rows_pos, rows_neg = [], []

        hma_scan_period = 55  # per request (fixed HMA 55 for this scanner)

        for sym in universe:
            try:
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                if close_full.empty:
                    continue

                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if close_show.empty or len(close_show) < max(10, hma_scan_period + 5):
                    continue

                hma_s = compute_hma(close_show, period=hma_scan_period).reindex(close_show.index).ffill()
                if hma_s.dropna().empty:
                    continue

                cross_up, _ = _cross_series(close_show, hma_s)
                cross_up = cross_up.reindex(close_show.index, fill_value=False)
                if not cross_up.any():
                    continue

                t_cross = cross_up[cross_up].index[-1]
                try:
                    loc = int(close_show.index.get_loc(t_cross))
                except Exception:
                    continue
                bars_since = int((len(close_show) - 1) - loc)
                if bars_since > int(hma_max_bars):
                    continue

                # Regression slope sign (uses existing helper; same daily_view range)
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue

                px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
                hma_cross = float(hma_s.loc[t_cross]) if (t_cross in hma_s.index and np.isfinite(hma_s.loc[t_cross])) else np.nan
                px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

                row = {
                    "Symbol": sym,
                    "Bars Since Cross": int(bars_since),
                    "Cross Time": t_cross,
                    "Price@Cross": px_cross,
                    "HMA(55)@Cross": hma_cross,
                    "Current Price": px_last,
                    "Regression Slope": float(m),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "AsOf": ts
                }

                if float(m) > 0.0:
                    rows_pos.append(row)
                elif float(m) < 0.0:
                    rows_neg.append(row)

            except Exception:
                continue

        left, right = st.columns(2)

        with left:
            st.subheader("(a) Regression > 0 — Cross ↑ HMA(55)")
            if not rows_pos:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Regression Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Regression < 0 — Cross ↑ HMA(55)")
            if not rows_neg:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Regression Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
# ---------------------------
# TAB 9: HMA SELL
# ---------------------------
with tab9:
    st.header("HMA Sell — Daily Price Cross ↓ HMA(55)")
    st.caption(
        "Lists symbols where **price recently crossed BELOW HMA(55)** on the **Daily** chart within the last **1–3 bars** "
        "(slider). Results are split into:\n"
        "• **(a) Regression > 0**\n"
        "• **(b) Regression < 0**\n\n"
        "Uses the selected **Daily view range** for the regression slope sign."
    )

    c1, c2 = st.columns(2)
    hma_sell_max_bars = c1.slider("Cross must be within last N bars", 1, 3, 2, 1, key="hma_sell_within_n")
    run_hma_sell = c2.button("Run HMA Sell Scan", key=f"btn_run_hma_sell_{mode}")

    if run_hma_sell:
        rows_pos, rows_neg = [], []
        hma_scan_period = 55  # fixed

        for sym in universe:
            try:
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                if close_full.empty:
                    continue

                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if close_show.empty or len(close_show) < max(10, hma_scan_period + 5):
                    continue

                hma_s = compute_hma(close_show, period=hma_scan_period).reindex(close_show.index).ffill()
                if hma_s.dropna().empty:
                    continue

                _, cross_dn = _cross_series(close_show, hma_s)
                cross_dn = cross_dn.reindex(close_show.index, fill_value=False)
                if not cross_dn.any():
                    continue

                t_cross = cross_dn[cross_dn].index[-1]
                try:
                    loc = int(close_show.index.get_loc(t_cross))
                except Exception:
                    continue
                bars_since = int((len(close_show) - 1) - loc)
                if bars_since > int(hma_sell_max_bars):
                    continue

                # Regression slope sign
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue

                px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
                hma_cross = float(hma_s.loc[t_cross]) if (t_cross in hma_s.index and np.isfinite(hma_s.loc[t_cross])) else np.nan
                px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

                row = {
                    "Symbol": sym,
                    "Bars Since Cross": int(bars_since),
                    "Cross Time": t_cross,
                    "Price@Cross": px_cross,
                    "HMA(55)@Cross": hma_cross,
                    "Current Price": px_last,
                    "Regression Slope": float(m),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "AsOf": ts
                }

                if float(m) > 0.0:
                    rows_pos.append(row)
                elif float(m) < 0.0:
                    rows_neg.append(row)

            except Exception:
                continue

        left, right = st.columns(2)

        with left:
            st.subheader("(a) Regression > 0 — Cross ↓ HMA(55)")
            if not rows_pos:
                st.info("No matches.")
            else:
                out_pos = pd.DataFrame(rows_pos)
                out_pos["Bars Since Cross"] = out_pos["Bars Since Cross"].astype(int)
                out_pos = out_pos.sort_values(["Bars Since Cross", "Regression Slope"], ascending=[True, False])
                st.dataframe(out_pos.reset_index(drop=True), use_container_width=True)

                csv = out_pos.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV (Regression > 0)",
                    data=csv,
                    file_name=f"hma_sell_reg_pos_{mode}.csv",
                    mime="text/csv",
                    key=f"dl_hma_sell_pos_{mode}"
                )

        with right:
            st.subheader("(b) Regression < 0 — Cross ↓ HMA(55)")
            if not rows_neg:
                st.info("No matches.")
            else:
                out_neg = pd.DataFrame(rows_neg)
                out_neg["Bars Since Cross"] = out_neg["Bars Since Cross"].astype(int)
                out_neg = out_neg.sort_values(["Bars Since Cross", "Regression Slope"], ascending=[True, True])
                st.dataframe(out_neg.reset_index(drop=True), use_container_width=True)

                csv = out_neg.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV (Regression < 0)",
                    data=csv,
                    file_name=f"hma_sell_reg_neg_{mode}.csv",
                    mime="text/csv",
                    key=f"dl_hma_sell_neg_{mode}"
                )

# ---------------------------
# TAB 10: HELP / NOTES
# ---------------------------
with tab10:
    st.header("Help / Notes")
    st.markdown(
        """
### What the scanners do
- **NTD -0.75 Scanner**: flags symbols whose latest **NTD** is below **-0.75** (hourly uses latest intraday; daily uses latest daily).
- **Recent BUY Scanner**: looks for a **daily NPX crossing above NTD** (green-circle condition) **and** a **positive** global daily trendline slope in the selected daily view range.
- **HMA Buy / Sell**: looks for **price crossing above/below HMA(55)** within the last **1–3 bars** (you choose), then splits results by the **sign** of the global regression slope (in the selected daily view range).

### Interpreting Regression Slope
- **Regression > 0**: the selected daily-range global trendline is upward.
- **Regression < 0**: the selected daily-range global trendline is downward.
- **R²**: goodness-of-fit of the global trendline over the selected range (higher generally means “cleaner” fit).

### Tips
- If you get few/no results, widen the **Daily view range** (more bars) or increase the “within last N bars” window.
- For very low-liquidity tickers, intraday data may be sparse; prefer **Daily** scans.

### Disclaimer
This tool is for analysis and visualization only and is not financial advice.
        """
    )
# ---------------------------
# TAB 9: HMA SELL
# ---------------------------
with tab9:
    st.header("HMA Sell — Daily Price Cross ↓ HMA(55)")
    st.caption(
        "Lists symbols where **price recently crossed BELOW HMA(55)** on the **Daily** chart within the last **1–3 bars** "
        "(slider). Results are split into:\n"
        "• **(a) Regression > 0**\n"
        "• **(b) Regression < 0**\n\n"
        "Uses the selected **Daily view range** for the regression slope sign."
    )

    c1, c2 = st.columns(2)
    hma_sell_max_bars = c1.slider("Cross must be within last N bars", 1, 3, 2, 1, key="hma_sell_within_n")
    run_hma_sell = c2.button("Run HMA Sell Scan", key=f"btn_run_hma_sell_{mode}")

    if run_hma_sell:
        rows_pos, rows_neg = [], []
        hma_scan_period = 55  # fixed

        for sym in universe:
            try:
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                if close_full.empty:
                    continue

                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if close_show.empty or len(close_show) < max(10, hma_scan_period + 5):
                    continue

                hma_s = compute_hma(close_show, period=hma_scan_period).reindex(close_show.index).ffill()
                if hma_s.dropna().empty:
                    continue

                _, cross_dn = _cross_series(close_show, hma_s)
                cross_dn = cross_dn.reindex(close_show.index, fill_value=False)
                if not cross_dn.any():
                    continue

                t_cross = cross_dn[cross_dn].index[-1]
                try:
                    loc = int(close_show.index.get_loc(t_cross))
                except Exception:
                    continue
                bars_since = int((len(close_show) - 1) - loc)
                if bars_since > int(hma_sell_max_bars):
                    continue

                # Regression slope sign
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue

                px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
                hma_cross = float(hma_s.loc[t_cross]) if (t_cross in hma_s.index and np.isfinite(hma_s.loc[t_cross])) else np.nan
                px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

                row = {
                    "Symbol": sym,
                    "Bars Since Cross": int(bars_since),
                    "Cross Time": t_cross,
                    "Price@Cross": px_cross,
                    "HMA(55)@Cross": hma_cross,
                    "Current Price": px_last,
                    "Regression Slope": float(m),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "AsOf": ts
                }

                if float(m) > 0.0:
                    rows_pos.append(row)
                elif float(m) < 0.0:
                    rows_neg.append(row)

            except Exception:
                continue

        left, right = st.columns(2)

        with left:
            st.subheader("(a) Regression > 0 — Cross ↓ HMA(55)")
            if not rows_pos:
                st.info("No matches.")
            else:
                out_pos = pd.DataFrame(rows_pos)
                out_pos["Bars Since Cross"] = out_pos["Bars Since Cross"].astype(int)
                out_pos = out_pos.sort_values(["Bars Since Cross", "Regression Slope"], ascending=[True, False])
                st.dataframe(out_pos.reset_index(drop=True), use_container_width=True)

                csv = out_pos.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV (Regression > 0)",
                    data=csv,
                    file_name=f"hma_sell_reg_pos_{mode}.csv",
                    mime="text/csv",
                    key=f"dl_hma_sell_pos_{mode}"
                )

        with right:
            st.subheader("(b) Regression < 0 — Cross ↓ HMA(55)")
            if not rows_neg:
                st.info("No matches.")
            else:
                out_neg = pd.DataFrame(rows_neg)
                out_neg["Bars Since Cross"] = out_neg["Bars Since Cross"].astype(int)
                out_neg = out_neg.sort_values(["Bars Since Cross", "Regression Slope"], ascending=[True, True])
                st.dataframe(out_neg.reset_index(drop=True), use_container_width=True)

                csv = out_neg.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV (Regression < 0)",
                    data=csv,
                    file_name=f"hma_sell_reg_neg_{mode}.csv",
                    mime="text/csv",
                    key=f"dl_hma_sell_neg_{mode}"
                )

# ---------------------------
# TAB 10: HELP / NOTES
# ---------------------------
with tab10:
    st.header("Help / Notes")
    st.markdown(
        """
### What the scanners do
- **NTD -0.75 Scanner**: flags symbols whose latest **NTD** is below **-0.75** (hourly uses latest intraday; daily uses latest daily).
- **Recent BUY Scanner**: looks for a **daily NPX crossing above NTD** (green-circle condition) **and** a **positive** global daily trendline slope in the selected daily view range.
- **HMA Buy / Sell**: looks for **price crossing above/below HMA(55)** within the last **1–3 bars** (you choose), then splits results by the **sign** of the global regression slope (in the selected daily view range).

### Interpreting Regression Slope
- **Regression > 0**: the selected daily-range global trendline is upward.
- **Regression < 0**: the selected daily-range global trendline is downward.
- **R²**: goodness-of-fit of the global trendline over the selected range (higher generally means “cleaner” fit).

### Tips
- If you get few/no results, widen the **Daily view range** (more bars) or increase the “within last N bars” window.
- For very low-liquidity tickers, intraday data may be sparse; prefer **Daily** scans.

### Disclaimer
This tool is for analysis and visualization only and is not financial advice.
        """
    )
