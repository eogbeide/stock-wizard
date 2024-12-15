import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import plotly.graph_objs as go

# Streamlit app title
st.title("Stock Price Forecasting with SARIMA and EMA")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA'])

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

    # Calculate 20-day and 200-day EMA
    ema_20 = data.ewm(span=20, adjust=False).mean()
    ema_200 = data.ewm(span=200, adjust=False).mean()

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

    # Step 5: Create the interactive plot using Plotly
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=data.index[-180:], y=data[-180:], mode='lines', name='Last 6 Months Historical Data', line=dict(color='blue')))

    # 20-day EMA
    fig.add_trace(go.Scatter(x=ema_20.index[-180:], y=ema_20[-180:], mode='lines', name='20-Day EMA', line=dict(color='red', dash='dash')))

    # 200-day EMA
    fig.add_trace(go.Scatter(x=ema_200.index[-180:], y=ema_200[-180:], mode='lines', name='200-Day EMA', line=dict(color='green', dash='dash')))

    # Forecast data
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='3 Months Forecast', line=dict(color='orange')))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=conf_int.iloc[:, 0],
        mode='lines',
        name='Lower Bound',
        line=dict(color='orange', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=conf_int.iloc[:, 1],
        mode='lines',
        name='Upper Bound',
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.3)',
        line=dict(color='orange', width=0),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(title=f'{ticker} Price Forecast for Next 3 Months',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      hovermode='x unified')

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
