import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Function to compute Bollinger Bands
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
    "For certain stocks, the direction of the 200-Day EMA indicates whether we are experiencing an upward or downward trend. "
    "A favorable buying opportunity during an upward trend arises when the closing price is near, at, or below the 200 EMA line. "
    "It’s even better if the Lower Bollinger Band is close to or touches the 200 EMA. Additionally, when the price crosses above "
    "the 30-day moving average, it indicates an upward trend and a potential buy signal."
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
    prices = data['Close'].asfreq('D').fillna(method='ffill')  # Set frequency to daily and forward fill

    # Check for NaN and infinite values
    if prices.isnull().any() or np.isinf(prices).any():
        st.error("Data contains NaN or infinite values. Please clean your data.")
        st.stop()

    # Ensure sufficient data points
    if len(prices) < 50:  # Adjust this threshold as needed
        st.error("Not enough data to fit the model.")
        st.stop()

    # Check for stationarity
    result = adfuller(prices.dropna())
    if result[1] > 0.05:  # If p-value > 0.05, the series is non-stationary
        st.warning("The time series is non-stationary. Consider differencing the series.")
        prices = prices.diff().dropna()  # Simple differencing

    # Calculate 200-day EMA
    ema_200 = prices.ewm(span=200, adjust=False).mean()

    # Calculate daily moving average (e.g., 30-day)
    moving_average = prices.rolling(window=30).mean()

    # Calculate Bollinger Bands
    lower_band, middle_band, upper_band = compute_bollinger_bands(prices)

    # Step 3: Fit the SARIMA model
    order = (1, 1, 1)  # Example values
    seasonal_order = (0, 0, 0, 0)  # Start with a simple seasonal order

    try:
        model = SARIMAX(prices, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
    except np.linalg.LinAlgError as e:
        st.error(f"LinAlgError: {str(e)}. Please check your data and model parameters.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

    # Step 4: Forecast the next one month (30 days)
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = forecast.predicted_mean

    # Get confidence intervals
    conf_int = forecast.conf_int()

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
    ax1.plot(upper_band[-360:], label='Bollinger Upper Band', color='purple', linestyle='--')  # Upper Bollinger Band

    # Get the current values
    current_ema_value = float(ema_200.iloc[-1])  # Current 200-day EMA
    current_lower_band_value = float(lower_band.iloc[-1])  # Current lower Bollinger Band
    current_upper_band_value = float(upper_band.iloc[-1])  # Current upper Bollinger Band
    current_moving_average_value = float(moving_average.iloc[-1])  # Current 30-Day MA
    current_close_value = float(prices.iloc[-1])  # Current Close price

    # Ensure that prices[-360:] is not empty and has enough data
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

    # Adjust y-axis limits to ensure the lines are visible
    ax1.set_ylim(bottom=min(price_min, current_lower_band_value) * 0.95, 
                  top=max(price_max, current_upper_band_value) * 1.05)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Move the legend to the bottom right corner with updated font size and style
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
