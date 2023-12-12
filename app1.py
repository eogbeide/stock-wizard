import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import glob
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import os

yf.pdr_override()

# Tickers list
ticker_list = ['INAB', 'CCCC', 'CADL', 'ADTX', 'MTCH', 'EA', 'PYPL', 'INTC', 'PFE', 'MRNA', 'VWAPY', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA', 'VTWG', 'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH', 'LU', 'META', 'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()

# We can get data by our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

files = []

def getData(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    SaveData(data, dataname)

# Create a data folder in your current dir.
def SaveData(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    df.to_csv(os.path.join(save_path, filename + '.csv'))

# This loop will iterate over ticker list, will pass one ticker to get data, and save that data as a file.
for tik in ticker_list:
    getData(tik)

# Pull data, train model, and predict
def select_file(files):
    num_files = len(files)

    while True:
        try:
            choice = st.sidebar.selectbox(
                "Select Company Ticker",
                range(1, num_files + 1),
                format_func=lambda x: files[x - 1].split('/')[-1].split('_')[0],
                key="selectbox"
            )
            selected_file = files[choice - 1]
            break
        except IndexError:
            st.sidebar.warning("Invalid choice. Please try again.")

    return selected_file

# the path to your csv file directory
mycsvdir = os.path.expanduser('~/Documents/data')

# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

# Prompt the user to select one file
selected_file = select_file(csvfiles)

# Read the selected file using pandas
df = pd.read_csv(selected_file)
df = df[['Date', 'Close']]
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df.reset_index(drop=True, inplace=True)

# Plot the selected file
title = f'Chart of Original Vs Predicted for ({selected_file.split("/")[-1].split("_")[0]})'

# Split the data into testing and training datasets
train = df[df['ds'] <= '10/31/2023']
test = df[df['ds'] >= '11/01/2023']

st.title("Major US Stocks Forecast Wizard")
st.write("")
st.subheader("The Smart Stock Trend Wiz: $$$")
st.write(selected_file.split("/")[-1].split("_")[0])
st.write("How to read chart: Below yhat_lower --> buy signal, above yhat_upper --> sell signal")
st.write(f"Number of days in train data: {len(train)}")
st.write(f"Number of days in test data: {len(test)}")

# Initialize Model
m = Prophet()

# Create and fit the prophet model to the training data
m.fit(train)

# Make predictions
future = m.make_future_dataframe(periods=93)
forecast = m.predict(future)

# Add predicted values to the original dataframe
df['predicted'] = forecast['trend']

# Plot the forecast and the original values for comparison
fig = px.line(df, x='ds', y=['y', 'predicted'], title=title)
fig.update_layout(xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# Delete the selected file
os.remove(selected_file.replace('\\', '/'))
