import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to fetch data
def fetch_data(symbol):
    data = yf.download(symbol, period="1y", interval="1d")
    return data['Volume']

# Function to fit SARIMA model and forecast
def sarima_forecast(data):
    # Fit the model
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    # Forecast the next 3 months
    forecast = results.get_forecast(steps=63)  # Approximately 63 trading days in 3 months
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    
    forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Forecast'])
    confidence_intervals = forecast.conf_int()
    return forecast_df, confidence_intervals

# Streamlit application
st.title("SPY Volume Forecasting with SARIMA")

# Dropdown to select stock symbol
symbol = st.selectbox("Select Symbol", ["SPY", "TSLA", "AMZN"])

# Fetch and display data
data = fetch_data(symbol)
st.write(f"Historical Volume Data for {symbol} (Last 12 Months):")
st.line_chart(data)

# Fit the SARIMA model and forecast
forecast_df, confidence_intervals = sarima_forecast(data)

# Plot the historical data and forecast
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data, label='Historical Volume', color='blue')
ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange')

# Confidence intervals
ax.fill_between(confidence_intervals.index,
                confidence_intervals.iloc[:, 0],
                confidence_intervals.iloc[:, 1],
                color='pink', alpha=0.3)

# Formatting the plot
ax.set_title('Stock Volume Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig)

# Display forecast data
#st.write("Forecast Data:")
#st.write(forecast_df)
