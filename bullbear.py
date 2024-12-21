# Step 5: Plot historical data, forecast, EMA, daily moving average, and Bollinger Bands
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot price and 200-day EMA
ax1.set_title(f'{ticker} Price Forecast, EMA, MA, and Bollinger Bands', fontsize=16)
ax1.plot(prices[-360:], label='Last 12 Months Historical Data', color='blue')  # Last 12 months of historical data
ax1.plot(ema_200[-360:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA for the last 12 months
ax1.plot(forecast_index, forecast_values, label='1 Month Forecast', color='orange')
ax1.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)

# Add daily moving average for the last 12 months
ax1.plot(moving_average[-360:], label='30-Day Moving Average', color='brown', linestyle='--')

# Plot Bollinger Bands
ax1.plot(lower_band[-360:], label='Bollinger Lower Band', color='red', linestyle='--')

# Get the current 200-day EMA value
current_ema_value = ema_200.iloc[-1]
current_date = prices.index[-1]

# Ensure current_ema_value is a float
current_ema_value = float(current_ema_value)

# Add a horizontal line for the current 200-day EMA price
ax1.axhline(y=current_ema_value, color='purple', linestyle='-', label='Current 200-Day EMA')

# Optional: Adjust y-axis limits to ensure the line is visible
ax1.set_ylim(bottom=min(prices[-360:].min(), current_ema_value) * 0.95, 
              top=max(prices[-360:].max(), current_ema_value) * 1.05)

ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

# Display the plot in Streamlit
st.pyplot(fig)
