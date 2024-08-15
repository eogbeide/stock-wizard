import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def load_data(ticker_symbol):
    spy_data = yf.Ticker(ticker_symbol)
    spy_history = spy_data.history(start="2001-01-01", actions=False)[["Close"]]
    spy_history = spy_history.reset_index()
    spy_history = spy_history.rename(columns={"Date": "ds", "Close": "y"})
    return spy_history

def main():
    st.title('Stock Price Forecasting with Prophet Model')

    ticker_symbol = st.sidebar.selectbox('Select Ticker Symbol', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'])
    
    data = load_data(ticker_symbol)

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.write('### Forecast Data')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    fig = model.plot(forecast, xlabel='Date', ylabel='Price')
    plt.title('Stock Price Forecast using Prophet Model')
    st.pyplot(fig)

if __name__ == '__main__':
    main()
