import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from plotly import graph_objs as go
import glob

from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
import os
from datetime import timedelta
from functools import lru_cache

yf.pdr_override()

# Tickers list
ticker_list = ['LLY', 'V', 'MA', 'ABBV', 'WBA', 'BMY', 'HUM', 'CI', 'UNH', 'CVS', 'DOCU', 'ZM', 'ABNB', 'SNOW', 'LYFT',
               'UBER', 'DLTR', 'DG', 'COST', 'KO', 'TGT', 'JNJ', 'HD', 'WMT', 'INAB', 'CCCC', 'CADL', 'ADTX', 'MTCH',
               'EA', 'PYPL', 'INTC', 'PFE', 'MRNA', 'CRL', 'CRM', 'AFRM', 'MU', 'AMAT', 'DELL', 'HPQ', 'BABA', 'VTWG',
               'SPGI', 'STX', 'LABU', 'TSM', 'AMZN', 'BOX', 'AAPL', 'NFLX', 'AMD', 'GME', 'GOOG', 'GUSH', 'LU', 'META',
               'MSFT', 'NVDA', 'PLTR', 'SITM', 'SPCE', 'SPY', 'TSLA', 'URI', 'WDC']
today = date.today()

# We can get data by our choice by giving days bracket
start_date = "2021-12-01"
end_date = today.strftime("%Y-%m-%d")  # Use today's date as the end date

# Get yesterday's date
yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

files = []


@lru_cache(maxsize=None)
def get_data(ticker):
    print(ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname = ticker + '_' + str(today)
    files.append(dataname)
    save_data(data, dataname)


# Create a data folder in your current dir.
def save_data(df, filename):
    save_path = os.path.expanduser('~/Documents/data/')
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    df.to_csv(os.path.join(save_path, filename + '.csv'))


# This loop will iterate over ticker list, will pass one ticker to get data, and save that data as a file.
for tik in ticker_list:
    get_data(tik)


def select_files(files):
    num_files = len(files)

    selected_files = []
    selected_ticker_info = None

    while True:
        try:
            choice = st.sidebar.selectbox(
                "Select Company Ticker",
                range(1, num_files + 1),
                format_func=lambda x: files[x - 1].split('/')[-1].split('_')[0],
                key="selectbox"
            )
            selected_file = files[choice - 1]
            selected_files.append(selected_file)

            # Retrieve ticker information from yfinance
            selected_ticker = selected_file.split('/')[-1].split('_')[0]
            ticker_info = yf.Ticker(selected_ticker)
            selected_ticker_info = ticker_info.info

            break
        except IndexError:
            st.sidebar.warning("Invalid choice. Please try again.")

    return selected_files, selected_ticker_info


# the path to your csv file directory
mycsvdir = os.path.expanduser('~/Documents/data')

# get all the csv files in that directory (assuming they have the extension .csv)
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

# Prompt the user to select two files
selected_files, selected_ticker_info = select_files(csvfiles)


@lru_cache(maxsize=None)
def read_csv_file(selected_file):
    df = pd.read_csv(selected_file)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df.reset_index(inplace=True, drop=True)
    return df


# Read the selected files using pandas
dfs = [read_csv_file(selected_file) for selected_file in selected_files]


titles = []
tickers = []
for selected_file in selected_files:
    ticker = selected_file.split('/')[-1].split```python
('/')[0]
    tickers.append(ticker)
    titles.append(ticker + ' Stock Price')


# Plot the stock prices using Plotly
fig = go.Figure()

for i, df in enumerate(dfs):
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name=tickers[i]))

fig.update_layout(
    title=','.join(titles),
    xaxis_title='Date',
    yaxis_title='Stock Price',
    xaxis_rangeslider_visible=True
)

st.plotly_chart(fig)

# Append today's date to the titles
today = date.today().strftime("%Y-%m-%d")

# Iterate over the selected files and their corresponding titles
for df, title, ticker in zip(dfs, titles, tickers):
    # Split the data into testing and training datasets
    train = df[df['ds'] <= '10/31/2023']
    test = df[df['ds'] >= '11/01/2023']

    st.title("Major US Stocks AI Forecast Wizard")
    st.write("")
    st.write("The Smart AI Stock Trend Wiz by Manny: $$$")
    st.write(f" - **Company Name:** ", selected_ticker_info['longName'])
    #st.write(" - Location: ", selected_ticker_info['country'])
    st.subheader("How to read chart:")
    st.write(f" - **yhat** is the median price that shows price trend")
    st.write(f" - **yhat_lower** is the lowest price. Actual price below yhat_lower signals a buying opportunity. Below yhat_lower --> Buy Signal")
    st.write(f" - **yhat_upper** is the highest price. Actual price above yhat_upper signals a selling or profit taking opportunity. Above yhat_upper --> Sell or Profit Taking Signal")
    #st.write(f"Number of months in train data for {ticker}: {len(train)}")
    #st.write(f" - Number of months in test data for {ticker}: {len(test)}")
    #st.write(f" - Number of days in train data: {len(train)}")
    #st.write(f" - Number of days in test data: {len(test)}")

    # Initialize Model
    m = Prophet()

    # Create and fit the prophet model to the training data
    m.fit(train)

    # Make predictions
    future = m.make_future_dataframe(periods=93)
    forecast = m.predict(future)
    #st.write("Forecast for", ticker)
    #   st.write(forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']].tail(30))

    # Add predicted values to the original dataframe
    df['predicted'] = forecast['trend']

    # Plot the forecast and the original values for comparison
    interactive_plot_forecasting(df, forecast, f'{title} ({today})')

    # Extract today's forecast values
    today_forecast = forecast[forecast['ds'] == today]

    # Get today's yhat, yhat_lower, and yhat_upper values
    today_yhat = round(today_forecast['yhat'].values[0],2)
    today_yhat_lower = round(today_forecast['yhat_lower'].values[0],2)
    today_yhat_upper = round(today_forecast['yhat_upper'].values[0],2)

    # Get yesterday's actual price
    yesterday_actual_price = round(df[df['ds'] == yesterday]['y'].values[0],2)


    # Check if yesterday's actual price exists
    st.subheader("Yesterday's Actual Price:")
    if yesterday in df['ds'].values:
        yesterday_actual_price = df[df['ds'] == yesterday]['y'].values[0]
    # Display today's forecast values
    if yesterday_actual_price is not None:
        st.write("- Yesterday's Price: ", yesterday_actual_price)
    st.subheader("Current Forecast Confidence Intervals:")
    st.write("- yhat: ", today_yhat)
    st.write("- yhat_lower: ", today_yhat_lower)
    st.write("- yhat_upper: ", today_yhat_upper)
    
# Delete existing files
for file in csvfiles:
    os.remove(file.replace('\\', '/'))

# Display selected ticker information
st.write("Selected Ticker Information:")
# st.write(selected_ticker_info)
st.subheader("Other Stats")
st.write(" - 50-Day Average: ", selected_ticker_info['fiftyDayAverage'])
st.write(" - 200-Day Average: ", selected_ticker_info['twoHundredDayAverage'])
#st.write(" - beta: ")
if 'beta' in selected_ticker_info:
    st.write(" - beta:", selected_ticker_info['beta'])
else:
    st.write(" - Beta Not Available")
st.subheader("About Company")
if 'longBusinessSummary' in selected_ticker_info:
    st.write(selected_ticker_info['longBusinessSummary'])
else:
    st.write("Not Available")

