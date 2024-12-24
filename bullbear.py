import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import plotly.graph_objects as go

# Function to calculate Bollinger Bands
def compute_bollinger_bands(data, window=20, num_sd=2):
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_sd)
    lower_band = middle_band - (std_dev * num_sd)
    return lower_band, middle_band, upper_band

# Streamlit app title
st.title("Stock Price Forecasting using SARIMA with EMA, MA & Bollinger")

# User input for stock ticker
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA'])

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Prepare the data
    prices = data['Close']
    prices.fillna(method='ffill', inplace=True)

    # Calculate Bollinger Bands
    lower_band, middle_band, upper_band = compute_bollinger_bands(prices)

    # Step 3: Fit the SARIMA model
    model = SARIMAX(prices, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Step 4: Forecast the next month
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = forecast.predicted_mean

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=forecast_index, y=upper_band[-1:], mode='lines', name='Upper Bollinger Band', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_index, y=lower_band[-1:], mode='lines', name='Lower Bollinger Band', line=dict(color='green', dash='dash')))

    # Update layout
    fig.update_layout(title=f'{ticker} Price Forecast', xaxis_title='Date', yaxis_title='Price')
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)
