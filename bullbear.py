import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import plotly.graph_objects as go

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
st.title("Stock Price Forecasting with SARIMA, EMA, RSI, and Bollinger Bands")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA', 'PLTR', 'NVDA', 'JYD', 'META', 'SITM', 'MARA', 'GOOG', 'HOOD', 'UBER', 'DOW', 'AFRM', 'MSFT', 'TSM', 'NFLX'])

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Prepare the data
    data = data['Close']  # Use the closing prices
    data = data.asfreq('D')  # Set frequency to daily
    data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values

    # Calculate 200-day EMA
    ema_200 = data.ewm(span=200, adjust=False).mean()

    # Calculate Bollinger Bands
    lower_band, middle_band, upper_band = compute_bollinger_bands(data)

    # Step 3: Fit the SARIMA model
    order = (1, 1, 1)  # Example values
    seasonal_order = (1, 1, 1, 12)  # Example values for monthly seasonality

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Step 4: Forecast the next three months (90 days)
    forecast_steps = 90
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = forecast.predicted_mean

    # Get confidence intervals
    conf_int = forecast.conf_int()

    # Step 5: Create an interactive plot using Plotly
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(x=data.index[-180:], y=data[-180:], mode='lines', name='Last 6 Months Historical Data', line=dict(color='blue')))
    
    # Add 200-day EMA
    fig.add_trace(go.Scatter(x=data.index[-180:], y=ema_200[-180:], mode='lines', name='200-Day EMA', line=dict(color='green', dash='dash')))
    
    # Add forecast data
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='3 Months Forecast', line=dict(color='orange')))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], mode='lines', name='Lower Bound', line=dict(color='orange', dash='dash'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], mode='lines', name='Upper Bound', line=dict(color='orange', dash='dash'), fill='tonexty'))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index[-180:], y=lower_band[-180:], mode='lines', name='Bollinger Lower Band', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index[-180:], y=upper_band[-180:], mode='lines', name='Bollinger Upper Band', line=dict(color='red', dash='dash')))

    # Update layout
    fig.update_layout(title=f'{ticker} Price Forecast, RSI, and Bollinger Bands', xaxis_title='Date', yaxis_title='Price', hovermode='x unified')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Create a DataFrame for forecast data including confidence intervals
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Price': forecast_values,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    })

    # Show the forecast data in a table
    st.write(forecast_df)
