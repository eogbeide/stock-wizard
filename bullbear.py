import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import plotly.graph_objects as go

# Streamlit app title
st.title("Stock Price Forecasting with SARIMA, EMA, and MACD")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA', 'PLTR', 'NVDA'])

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

    # Calculate MACD
    short_ema = data.ewm(span=12, adjust=False).mean()  # Short-term EMA
    long_ema = data.ewm(span=26, adjust=False).mean()  # Long-term EMA
    macd_line = short_ema - long_ema  # MACD Line
    signal_line = macd_line.ewm(span=9, adjust=False).mean()  # Signal Line

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

    # Step 5: Create interactive plot using Plotly
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(x=data.index[-180:], y=data[-180:], mode='lines', name='Last 6 Months Historical Data', line=dict(color='blue')))
    
    # Add 200-day EMA
    fig.add_trace(go.Scatter(x=data.index[-180:], y=ema_200[-180:], mode='lines', name='200-Day EMA', line=dict(color='green', dash='dash')))
    
    # Add forecasted data
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='3 Months Forecast', line=dict(color='orange')))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], mode='lines', name='Lower Bound', line=dict(color='orange', dash='dash'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], mode='lines', name='Upper Bound', line=dict(color='orange', dash='dash'), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.3)', showlegend=False))

    # Add MACD line
    fig.add_trace(go.Scatter(x=data.index[-180:], y=macd_line[-180:], mode='lines', name='MACD Line', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=data.index[-180:], y=signal_line[-180:], mode='lines', name='Signal Line', line=dict(color='red', dash='dash')))

    # Update layout
    fig.update_layout(title=f'{ticker} Price Forecast and MACD', xaxis_title='Date', yaxis_title='Price/MACD', template='plotly_white')

    # Display the interactive plot in Streamlit
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
