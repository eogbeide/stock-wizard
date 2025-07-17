# --- Tab 1: Original US Forecast ---
with tab1:
    st.header("🇺🇸 Original US Forecast")
    if mode == "Forex":
        pair = st.selectbox(
            "Select Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'],
            key="orig_forex_pair"
        )
        chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_forex_chart")
        if st.button("Run Forex Forecast", key="orig_forex_btn"):
            # Daily series
            df = yf.download(pair, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                   .asfreq("D").fillna(method="ffill")
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)
            rsi    = compute_rsi(df)

            # Forecast
            model = safe_sarimax(df, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx   = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
            vals, ci = fc.predicted_mean, fc.conf_int()

            if chart in ("Daily","Both"):
                # Price + BB + EMA + MA + Forecast
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                ax.set_title(f"{pair} Daily Forecast")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

                # RSI panel
                fig_rsi, ax_rsi = plt.subplots(figsize=(14,2))
                ax_rsi.plot(rsi[-360:], label="RSI(14)")
                ax_rsi.axhline(70, linestyle="--")
                ax_rsi.axhline(30, linestyle="--")
                ax_rsi.set_title("RSI (14)")
                ax_rsi.legend(loc="lower left")
                st.pyplot(fig_rsi)

            if chart in ("Hourly","Both"):
                # Intraday series
                intraday = yf.download(pair, period="1d", interval="5m")
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    hc = intraday["Close"].ffill()
                    he = hc.ewm(span=20).mean()
                    rsi_i = compute_rsi(hc)

                    # Price + 20 EMA
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label="Intraday")
                    ax2.plot(he, "--", label="20 EMA")
                    ax2.set_title(f"{pair} Intraday (5m)")
                    ax2.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig2)

                    # Intraday RSI
                    fig2_rsi, ax2_rsi = plt.subplots(figsize=(14,2))
                    ax2_rsi.plot(rsi_i, label="RSI(14)")
                    ax2_rsi.axhline(70, linestyle="--")
                    ax2_rsi.axhline(30, linestyle="--")
                    ax2_rsi.set_title("Intraday RSI (14)")
                    ax2_rsi.legend(loc="lower left")
                    st.pyplot(fig2_rsi)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))


# --- Tab 2: Enhanced US Forecast ---
with tab2:
    st.header("🇺🇸 Enhanced US Forecast")
    if mode == "Forex":
        pair = st.selectbox(
            "Select Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'],
            key="enh_forex_pair"
        )
        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_forex_view")
        if st.button("Run Enhanced Forex Forecast", key="enh_forex_btn"):
            # Daily
            daily = yf.download(pair, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                       .asfreq("D").fillna(method="ffill")
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi     = compute_rsi(daily)

            model = safe_sarimax(daily, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx, vals, ci = (
                pd.date_range(daily.index[-1] + timedelta(1), periods=30, freq="D"),
                fc.predicted_mean,
                fc.conf_int()
            )

            if view in ("Daily","Both"):
                # Price + BB + EMA + Fib
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                ax.set_title(f"{pair} Daily + Fib")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

                # Daily RSI
                fig_rsi, ax_rsi = plt.subplots(figsize=(14,2))
                ax_rsi.plot(rsi[-360:], label="RSI(14)")
                ax_rsi.axhline(70, linestyle="--")
                ax_rsi.axhline(30, linestyle="--")
                ax_rsi.set_title("RSI (14)")
                ax_rsi.legend(loc="lower left")
                st.pyplot(fig_rsi)

            if view in ("Intraday","Both"):
                intraday = yf.download(pair, period="1d", interval="5m")
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    ic = intraday["Close"].ffill()
                    ie = ic.ewm(span=20).mean()
                    rsi_i = compute_rsi(ic)

                    # Intraday price + BB
                    fig3, ax3 = plt.subplots(figsize=(14,5))
                    ax3.plot(ic, label="Intraday")
                    ax3.plot(ie, "--", label="20 EMA")
                    ax3.plot(*compute_bollinger_bands(ic), "--", label=["Lower BB","Upper BB"])
                    ax3.set_title(f"{pair} Intraday + Fib")
                    ax3.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig3)

                    # Intraday RSI
                    fig4, ax4 = plt.subplots(figsize=(14,2))
                    ax4.plot(rsi_i, label="RSI(14)")
                    ax4.axhline(70, linestyle="--")
                    ax4.axhline(30, linestyle="--")
                    ax4.set_title("Intraday RSI (14)")
                    ax4.legend(loc="lower left")
                    st.pyplot(fig4)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))
