# Calculate support and resistance levels
    support_level = prices.rolling(window=30).min().iloc[-1]  # Lowest price in the last 30 days
    resistance_level = prices.rolling(window=30).max().iloc[-1]  # Highest price in the last 30 days

    # Ensure support and resistance levels are not NaN
    if pd.isna(support_level) or pd.isna(resistance_level):
        st.error("Unable to calculate support and resistance levels. Please check the data.")
    else:
        # Step 5: Plot historical data, forecast, EMA, daily moving average, Bollinger Bands, support and resistance levels
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot price and 200-day EMA
        ax1.set_title(f'{ticker} Price Forecast, EMA, MA, Bollinger Bands, Support & Resistance', fontsize=16)
        ax1.plot(prices[-360:], label='Last 12 Months Historical Data', color='blue')  # Last 12 months of historical data
        ax1.plot(ema_200[-360:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA
        ax1.plot(forecast_index, forecast_values, label='1 Month Forecast', color='orange')
        ax1.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)

        # Add daily moving average for the last 12 months
        ax1.plot(moving_average[-360:], label='30-Day Moving Average', color='brown', linestyle='--')

        # Plot Bollinger Bands
        ax1.plot(lower_band[-360:], label='Bollinger Lower Band', color='red', linestyle='--')

        # Plot support and resistance levels if valid
        ax1.axhline(y=support_level, color='green', linestyle='--', label='Support Level')
        ax1.axhline(y=resistance_level, color='red', linestyle='--', label='Resistance Level')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Display the plot in Streamlit
        st.pyplot(fig)
