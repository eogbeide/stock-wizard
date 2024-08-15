import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(ticker_symbol):
    spy_data = yf.Ticker(ticker_symbol)
    spy_history = spy_data.history(start="2001-01-01", actions=False)[["Close"]]
    return spy_history

def create_LSTM_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(30, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    st.title('Stock Price Forecasting with LSTM Model')

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']  # List of ticker symbols
    ticker_symbol = st.sidebar.selectbox('Select Ticker Symbol', tickers)

    data = load_data(ticker_symbol)
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(30, len(dataset)):
        x_train.append(scaled_data[i-30:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_LSTM_model()
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    st.write("LSTM model trained and ready for forecasting.")

if __name__ == '__main__':
    main()
