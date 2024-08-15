import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def load_data(ticker_symbol):
    spy_data = yf.Ticker(ticker_symbol)
    # Define the ticker symbol for SPY
    ticker_symbol = (['DOW','SPY', 'META', 'TSLA', 'AMZN', 'GOOG', 'UNH', 'SPCE', 'NVDA'])
    spy_history = spy_data.history(period="1y")[["Open", "High", "Low", "Close"]]
    return spy_history["Close"]

def main():
    st.title('Stock Forecasting App')

    ticker_symbol = st.sidebar.text_input('Enter Ticker Symbol', 'AAPL')
    data = load_data(ticker_symbol)

    st.write(f"### {ticker_symbol} Stock Data")
    st.write(data)

    model = ARIMA(data, order=(2, 1, 2))
    model_fit = model.fit()

    forecast, stderr, conf_int = model_fit.forecast(steps=30, alpha=0.05)

    st.write("### Forecasted Stock Prices")
    forecast_df = pd.DataFrame({
        "Date": pd.date_range(start=data.index[-1], periods=31)[1:],
        "Forecasted Price": forecast,
        "Lower CI": forecast - 1.96 * stderr,
        "Upper CI": forecast + 1.96 * stderr
    })
    st.write(forecast_df)

    st.write("### Interactive Plot")
    fig, ax = plt.subplots()
    ax.plot(data.index, data, label='Actual')
    ax.plot(forecast_df["Date"], forecast_df["Forecasted Price"], label='Forecast', color='green')
    ax.fill_between(forecast_df["Date"], forecast_df["Lower CI"], forecast_df["Upper CI"], color='lightgray', label='Confidence Intervals')
    ax.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()
