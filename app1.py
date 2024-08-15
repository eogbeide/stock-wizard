import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet

def load_data(ticker_symbol):
    spy_data = yf.Ticker(ticker_symbol)
    spy_history = spy_data.history(start="2001-01-01", actions=False)[["Close"]]
    
    spy_history.reset_index(inplace=True)
    spy_history.columns = ['ds', 'y']  # Renaming columns for Prophet
    
    return spy_history

def main():
    st.title('Stock Price Forecasting with Prophet')

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']  # List of ticker symbols
    ticker_symbol = st.sidebar.selectbox('Select Ticker Symbol', tickers)

    final_df = load_data(ticker_symbol)

    model = Prophet()
    model.fit(final_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(final_df['ds'], final_df['y'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.legend()

    st.pyplot(fig)

    st.write("Forecast:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))  # Display forecast for the last 30 days

if __name__ == '__main__':
    main()
