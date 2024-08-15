import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Define the ticker symbol for SPY
ticker_symbol = "SPY"

# Fetch historical data for SPY from Yahoo Finance starting from 2001
spy_data = yf.Ticker(ticker_symbol)

# Get historical market data for the specified columns since 2001
spy_history = spy_data.history(start="2001-01-01", actions=False)[["Open", "High", "Low", "Close"]]

# Convert 'Date' column to date data type
spy_history.index = spy_history.index.date

# Create a final dataframe with only the "Close" column
final_df = spy_history[["Close"]]

# Train the ARIMA model
model = ARIMA(final_df["Close"], order=(2, 1, 2))  # Example order, adjust as needed
model_fit = model.fit()

# Validate the model on a validation set (e.g., last 30 days)
n_train = len(final_df) - 30
train = final_df["Close"][:n_train]
test = final_df["Close"][n_train:]

# Make predictions for the validation set
predictions = model_fit.predict(start=n_train, end=len(final_df)-1, typ='levels')

# Calculate MAPE for the validation set
mape = np.mean(np.abs((test - predictions) / test)) * 100

print(f"Mean Absolute Percentage Error (MAPE) on the validation set: {mape:.2f}%")

# Forecast the next 30 days
forecast = model_fit.forecast(steps=30)

# Plot the predictions and forecast
plt.figure(figsize=(12, 6))
plt.plot(final_df.index, final_df["Close"], label='Actual')
plt.plot(predictions.index, predictions, label='Predictions', color='red')
plt.plot(np.arange(len(final_df), len(final_df) + 30), forecast, label='Forecast', color='green')
plt.legend()
plt.show()
