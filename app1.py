import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def load_data(ticker_symbol):
    spy_data = yf.Ticker(ticker_symbol)
    spy_history = spy_data.history(start="2001-01-01", actions=False)[["Open", "High", "Low", "Close"]]
    
    # Convert datetime index to date datatype
    spy_history.index = pd.to_datetime(spy_history.index).date
    spy_history['Date'] = spy_history.index  # No need to convert datetime to date 
        
    final_df = spy_history[["Close"]]
    
    # Set index and column names to None
    final_df.index.name = None
    final_df.columns.name = None
    
    return final_df

def main():
    st.title('Stock Price Forecasting with ARIMA Model')

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']  # List of ticker symbols
    ticker_symbol = st.sidebar.selectbox('Select Ticker Symbol', tickers)

    final_df = load_data(ticker_symbol)

    order_p = st.sidebar.slider('Order p', 0, 10, 2)
    order_d = st.sidebar.slider('Order d', 0, 10, 1)
    order_q = st.sidebar.slider('Order q', 0, 10, 2)

    model = ARIMA(final_df["Close"], order=(order_p, order_d, order_q))
    model_fit = model.fit()

    n_train = len(final_df) - 90
    train = final_df["Close"][:n_train]
    test = final_df["Close"][n_train:]

    predictions = model_fit.predict(start=n_train, end=len(final_df) - 1, typ='levels')

    mape = np.mean(np.abs((test - predictions) / test)) * 100

    st.write(f"Mean Absolute Percentage Error (MAPE) on the test set: {mape:.2f}%")

    forecast = model_fit.forecast(steps=90)
    forecast_dates = pd.date_range(final_df.index[-1], periods=91)[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    
    # Set index and column names to None for forecast_df
    forecast_df.index.name = None
    forecast_df.columns.name = None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(final_df.index, final_df["Close"], label='Actual')
    ax.plot(predictions.index, predictions, label='Predictions', color='red')
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='green')
    ax.legend()

    st.pyplot(fig)

    st.write("Predictions:")    
    st.write(forecast_df[['Date', 'Forecast']].head(10).set_index('Date').round(2))  # Display only 'Date' and 'Forecast' columns

if __name__ == '__main__':
    main()
