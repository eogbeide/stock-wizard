# Step 5: Plot historical data, forecast, and EMA
plt.figure(figsize=(14, 7))
plt.plot(data[-180:], label='Last 6 Months Historical Data', color='blue')  # Last 6 months of historical data
plt.plot(ema_20[-180:], label='20-Day EMA', color='red', linestyle='--')  # 20-day EMA
plt.plot(ema_200[-180:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA
plt.plot(forecast_index, forecast_values, label='3 Months Forecast', color='orange')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)

# Ensure values are computed correctly
last_forecast_value = forecast_values.iloc[-1]
last_ema_20_value = ema_20.iloc[-1] if not ema_20.empty else np.nan
last_ema_200_value = ema_200.iloc[-1] if not ema_200.empty else np.nan

# Adding the last value annotations
plt.annotate(f'Forecast: {last_forecast_value:.2f}', 
             xy=(forecast_index[-1], last_forecast_value), 
             xytext=(forecast_index[-1], last_forecast_value + 5),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

if not np.isnan(last_ema_20_value):
    plt.annotate(f'20-Day EMA: {last_ema_20_value:.2f}', 
                 xy=(data.index[-1], last_ema_20_value), 
                 xytext=(data.index[-1], last_ema_20_value + 5),
                 arrowprops=dict(facecolor='red', arrowstyle='->'))

if not np.isnan(last_ema_200_value):
    plt.annotate(f'200-Day EMA: {last_ema_200_value:.2f}', 
                 xy=(data.index[-1], last_ema_200_value), 
                 xytext=(data.index[-1], last_ema_200_value + 5),
                 arrowprops=dict(facecolor='green', arrowstyle='->'))

plt.title(f'{ticker} Price Forecast for Next 3 Months')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)
