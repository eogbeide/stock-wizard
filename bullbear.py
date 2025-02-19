import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt

# Function to compute RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def compute_bollinger_bands(data, window=20, num_sd=2):
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_sd)
    lower_band = middle_band - (std_dev * num_sd)
    return lower_band, middle_band, upper_band

# Streamlit app title
st.title("Stock Price Forecasting using SARIMA with EMA, MA & Bollinger")

# Information box at the top
st.info(
    "For certain stocks, the direction of the 200-Day EMA indicates whether we are experiencing an upward or downward trend."
)

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=sorted([
    'AAPL', 'SPY', 'AMZN', 'TSLA', 'SOFI', 'TSM', 'JPM', 'SPHD', 'VTSAX', 'HDV',
    'NVDA', 'META', 'SITM', 'SPGI', 'JYD', 'AVGO', 'PG', 'VTWAX', 'VIG',
    'MARA', 'GOOG', 'HOOD', 'BABA', 'SMR', 'MA', 'VYM', 'VONE', 'QQQM',
    'MSFT', 'DIA', 'NFLX', 'URI', 'VOO', 'BAC', 'BJ', 'FNILX', 'RSP'
]))

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Prepare the data
    prices = data['Close']  
    prices = prices.asfreq('D')  
    prices.fillna(method='ffill', inplace=True)  

    # Check if there is enough data
    if len(prices) < 50:  # Adjust this threshold as needed
        st.error("Not enough data to fit the model.")
    else:
        # Calculate 200-day EMA
        ema_200 = prices.ewm(span=200, adjust=False).mean()

        # Calculate daily moving average (e.g., 30-day)
        moving_average = prices.rolling(window=30).mean()

        # Calculate Bollinger Bands
        lower_band, middle_band, upper_band = compute_bollinger_bands(prices)

        # Step 3: Fit the SARIMA model
        order = (1, 1, 1)  
        seasonal_order = (1, 1, 1, 12)  

        try:
            model = SARIMAX(prices, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
        except np.linalg.LinAlgError:
            st.error("Matrix is singular. Please check your data and model parameters.")
            st.stop()

        # Step 4: Forecast the next month (30 days)
        forecast_steps = 30
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_values = forecast.predicted_mean

        # Get confidence intervals
        conf_int = forecast.conf_int()

        # Step 5: Plot historical data, forecast, EMA, daily moving average, and Bollinger Bands
        fig, ax1 = plt.subplots(figsize=(14, 7))

        ax1.set_title(f'{ticker} Price Forecast, EMA, MA, and Bollinger Bands', fontsize=16)
        ax1.plot(prices[-360:], label='Last 12 Months Historical Data', color='blue')
        ax1.plot(ema_200[-360:], label='200-Day EMA', color='green', linestyle='--')
        ax1.plot(forecast_index, forecast_values, label='1 Month Forecast', color='orange')
        ax1.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
        ax1.plot(moving_average[-360:], label='30-Day Moving Average', color='brown', linestyle='--')
        ax1.plot(lower_band[-360:], label='Bollinger Lower Band', color='red', linestyle='--')
        ax1.plot(upper_band[-360:], label='Bollinger Upper Band', color='purple', linestyle='--')

        # Current values
        current_ema_value = float(ema_200.iloc[-1])
        current_lower_band_value = float(lower_band.iloc[-1])
        current_upper_band_value = float(upper_band.iloc[-1])
        current_moving_average_value = float(moving_average.iloc[-1])
        current_close_value = float(prices.iloc[-1])

        if len(prices) > 360:
            price_min = float(prices[-360:].min())
            price_max = float(prices[-360:].max())
        else:
            price_min = float(prices.min())
            price_max = float(prices.max())

        # Add horizontal lines for the current values
        ax1.axhline(y=current_upper_band_value, color='purple', linestyle='-', label=f'Current Upper Bollinger Band: {current_upper_band_value:.2f}')
        ax1.axhline(y=current_moving_average_value, color='brown', linestyle='-', label=f'Current 30-Day MA: {current_moving_average_value:.2f}')
        ax1.axhline(y=current_close_value, color='blue', linestyle='-', label=f'Current Close Price: {current_close_value:.2f}')
        ax1.axhline(y=current_lower_band_value, color='red', linestyle='-', label=f'Current Lower Bollinger Band: {current_lower_band_value:.2f}')
        ax1.axhline(y=current_ema_value, color='green', linestyle='-', label=f'Current 200-Day EMA: {current_ema_value:.2f}') 

        ax1.set_ylim(bottom=min(price_min, current_lower_band_value) * 0.95, 
                      top=max(price_max, current_upper_band_value) * 1.05)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax1.legend(loc='lower right', fontsize='x-small', fancybox=True, framealpha=0.5, title='Legend', title_fontsize='small')

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Create a DataFrame for forecast data including confidence intervals
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted Price': forecast_values,
            'Lower Bound': conf_int.iloc[:, 0],
            'Upper Bound': conf_int.iloc[:, 1]
        })

        # Show the forecast data in a table
        st.write(forecast_df)
