import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# Streamlit app title
st.title("Stock Price Forecasting with SARIMA, EMA, and MACD")

# Define stock tickers
tickers = ['AAPL', 'SPY', 'AMZN', 'TSLA', 'PLTR', 'NVDA', 'JYD', 'META', 'SITM', 'MARA', 
           'GOOG', 'HOOD', 'UBER', 'DOW', 'AFRM', 'MSFT', 'TSM', 'NFLX']

# Create two columns for two different stock forecasts
col1, col2 = st.columns(2)

# Define common start and end dates
start_date = '2018-01-01'
end_date = pd.to_datetime("today")

# Column 1: First stock selection and forecast
with col1:
    ticker1 = st.selectbox("Select First Stock Ticker:", options=tickers)
    if st.button("Forecast First Stock"):
        # Download historical data from Yahoo Finance
        data1 = yf.download(ticker1, start=start_date, end=end_date)

        # Prepare the data
        data1 = data1['Close'].asfreq('D').fillna(method='ffill')

        # Calculate 200-day EMA
        ema_200_1 = data1.ewm(span=200, adjust=False).mean()

        # Calculate MACD
        short_ema_1 = data1.ewm(span=12, adjust=False).mean()
        long_ema_1 = data1.ewm(span=26, adjust=False).mean()
        macd_line_1 = short_ema_1 - long_ema_1
        signal_line_1 = macd_line_1.ewm(span=9, adjust=False).mean()

        # Fit the SARIMA model
        model1 = SARIMAX(data1, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit_1 = model1.fit(disp=False)

        # Forecast the next three months (90 days)
        forecast_steps = 90
        forecast_1 = model_fit_1.get_forecast(steps=forecast_steps)
        forecast_index_1 = pd.date_range(start=data1.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_values_1 = forecast_1.predicted_mean
        conf_int_1 = forecast_1.conf_int()

        # Plotting
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        ax1.set_title(f'{ticker1} Price Forecast and MACD', fontsize=16)
        ax1.plot(data1[-180:], label='Last 6 Months Historical Data', color='blue')
        ax1.plot(ema_200_1[-180:], label='200-Day EMA', color='green', linestyle='--')
        ax1.plot(forecast_index_1, forecast_values_1, label='3 Months Forecast', color='orange')
        ax1.fill_between(forecast_index_1, conf_int_1.iloc[:, 0], conf_int_1.iloc[:, 1], color='orange', alpha=0.3)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Create a second y-axis for MACD
        ax2 = ax1.twinx()  
        ax2.plot(macd_line_1[-180:], label='MACD Line', color='purple')
        ax2.plot(signal_line_1[-180:], label='Signal Line', color='red', linestyle='--')
        ax2.set_ylabel('MACD', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')

        # Add MACD legend
        ax2.legend(loc='upper right')

        # Display the plot in Streamlit
        st.pyplot(fig1)

        # Create a DataFrame for forecast data including confidence intervals
        forecast_df_1 = pd.DataFrame({
            'Date': forecast_index_1,
            'Forecasted Price': forecast_values_1,
            'Lower Bound': conf_int_1.iloc[:, 0],
            'Upper Bound': conf_int_1.iloc[:, 1]
        })

        # Show the forecast data in a table
        st.write(forecast_df_1)

# Column 2: Second stock selection and forecast
with col2:
    ticker2 = st.selectbox("Select Second Stock Ticker:", options=tickers)
    if st.button("Forecast Second Stock"):
        # Download historical data from Yahoo Finance
        data2 = yf.download(ticker2, start=start_date, end=end_date)

        # Prepare the data
        data2 = data2['Close'].asfreq('D').fillna(method='ffill')

        # Calculate 200-day EMA
        ema_200_2 = data2.ewm(span=200, adjust=False).mean()

        # Calculate MACD
        short_ema_2 = data2.ewm(span=12, adjust=False).mean()
        long_ema_2 = data2.ewm(span=26, adjust=False).mean()
        macd_line_2 = short_ema_2 - long_ema_2
        signal_line_2 = macd_line_2.ewm(span=9, adjust=False).mean()

        # Fit the SARIMA model
        model2 = SARIMAX(data2, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit_2 = model2.fit(disp=False)

        # Forecast the next three months (90 days)
        forecast_2 = model_fit_2.get_forecast(steps=forecast_steps)
        forecast_index_2 = pd.date_range(start=data2.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_values_2 = forecast_2.predicted_mean
        conf_int_2 = forecast_2.conf_int()

        # Plotting
        fig2, ax1 = plt.subplots(figsize=(14, 7))
        ax1.set_title(f'{ticker2} Price Forecast and MACD', fontsize=16)
        ax1.plot(data2[-180:], label='Last 6 Months Historical Data', color='blue')
        ax1.plot(ema_200_2[-180:], label='200-Day EMA', color='green', linestyle='--')
        ax1.plot(forecast_index_2, forecast_values_2, label='3 Months Forecast', color='orange')
        ax1.fill_between(forecast_index_2, conf_int_2.iloc[:, 0], conf_int_2.iloc[:, 1], color='orange', alpha=0.3)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Create a second y-axis for MACD
        ax2 = ax1.twinx()  
        ax2.plot(macd_line_2[-180:], label='MACD Line', color='purple')
        ax2.plot(signal_line_2[-180:], label='Signal Line', color='red', linestyle='--')
        ax2.set_ylabel('MACD', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')

        # Add MACD legend
        ax2.legend(loc='upper right')

        # Display the plot in Streamlit
        st.pyplot(fig2)

        # Create a DataFrame for forecast data including confidence intervals
        forecast_df_2 = pd.DataFrame({
            'Date': forecast_index_2,
            'Forecasted Price': forecast_values_2,
            'Lower Bound': conf_int_2.iloc[:, 0],
            'Upper Bound': conf_int_2.iloc[:, 1]
        })

        # Show the forecast data in a table
        st.write(forecast_df_2)
